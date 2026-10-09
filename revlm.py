"""A ~20M-parameter GPT with a standard residual stack or a reversible one.

Variants of the layer stack (h = step size):

  standard   x <- x + Attn(LN1 x);  x <- x + MLP(LN2 x)                     (autograd stores everything)
  midpoint   p[l+1] = p[l-1] + 2h f(p[l])                                     Gal et al. 2025, Eq. 4
  leapfrog   p[l+1] = 2 p[l] - p[l-1] + h^2 f(p[l])                           Eq. 6
  blend      p[l+1] = a p[l-1] + (1-a) p[l] + h f(p[l])                       Eq. 15 with a fixed a ("midpoint (a)")
  euler      q[l+1] = q[l] + h Attn(LN1 p[l]);  p[l+1] = p[l] + h MLP(LN2 q[l+1])
             (the Hamiltonian / symplectic-Euler two-stream form, Eqs. 8-9 with a = b = 1)

where f(p) = Attn(LN1 p) + MLP(LN2(p + Attn(LN1 p))) (Eq. 5).

The reversible variants run the whole stack inside one autograd.Function. The forward pass keeps only the
final pair of states; the backward pass walks the layers in reverse, rebuilds each layer's input from its
output by running the update backwards, and re-runs that layer once with autograd on to get its gradients.
"""
import json
import math
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")

# Mixed precision is off by default: on the GTX 1660 Ti (TU116, no tensor cores) cuBLAS fp16 matmuls measured
# 0.46 TFLOPS against 4.1 TFLOPS in fp32, so fp32 training is several times faster on this card. On a GPU that
# does have tensor cores the answer flips, so `colab_env.choose_precision()` measures both and sets these two.
AMP = False
AMP_DTYPE = torch.float16


def limit_to_dedicated_vram(margin_mib=200):
    """Windows (WDDM) lets CUDA spill into shared system RAM instead of raising OOM, which makes an oversized batch
    crawl rather than fail. Cap PyTorch's allocator at the dedicated VRAM that is free now, so OOM is a real OOM."""
    free, total = torch.cuda.mem_get_info()
    frac = max(0.1, (free - margin_mib * 2**20) / total)
    torch.cuda.set_per_process_memory_fraction(frac)
    return frac * total / 2**20


def amp_ctx():
    return torch.autocast("cuda", dtype=AMP_DTYPE, enabled=AMP)


# coefficients of the generic two-step rule v = cs*s + cu*u + k*f(u)
TWO_STEP = {
    "midpoint": lambda h, a: (1.0, 0.0, 2 * h),
    "leapfrog": lambda h, a: (-1.0, 2.0, h * h),
    "blend": lambda h, a: (a, 1.0 - a, h),
}
VARIANTS = ["standard", "checkpoint", "euler", *TWO_STEP]


# ----------------------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.attn_out = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.mlp_out = nn.Linear(4 * d, d, bias=False)

    def attn(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.attn_out(y.transpose(1, 2).reshape(B, T, C))

    def mlp(self, x):
        return self.mlp_out(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))

    def f(self, p):  # whole block as one function of the state (Eq. 5)
        a = self.attn(p)
        return a + self.mlp(p + a)

    def attn_params(self):
        return [*self.ln1.parameters(), *self.qkv.parameters(), *self.attn_out.parameters()]

    def mlp_params(self):
        return [*self.ln2.parameters(), *self.fc.parameters(), *self.mlp_out.parameters()]


def _euler_step(blk, q, p, h):
    q = q + h * blk.attn(p).float()
    p = p + h * blk.mlp(q).float()
    return q, p


def _two_step(blk, s, u, coef):
    cs, cu, k = coef
    v = cs * s + k * blk.f(u).float()
    if cu:
        v = v + cu * u
    return u, v


class RevStack(torch.autograd.Function):
    """Runs the reversible stack; stores only the final (state0, state1) pair."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x0, x1, variant, h, alpha, blocks, *params):
        ctx.variant, ctx.h, ctx.alpha, ctx.blocks = variant, h, alpha, blocks
        with torch.no_grad():
            a, b = x0, x1
            for blk in blocks:
                if variant == "euler":
                    a, b = _euler_step(blk, a, b, h)
                else:
                    a, b = _two_step(blk, a, b, TWO_STEP[variant](h, alpha))
        ctx.save_for_backward(a, b)
        return a, b

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, ga, gb):
        a, b = (t.detach() for t in ctx.saved_tensors)
        ga = torch.zeros_like(a) if ga is None else ga
        gb = torch.zeros_like(b) if gb is None else gb
        h, grads = ctx.h, {}

        def acc(ps, gs):
            for p, g in zip(ps, gs):
                if g is not None:
                    grads[p] = g if p not in grads else grads[p] + g

        for blk in reversed(ctx.blocks):
            with torch.enable_grad():
                if ctx.variant == "euler":
                    # forward was q1 = q0 + h A(p0); p1 = p0 + h M(q1).  (a, b) = (q1, p1)
                    q1 = a.detach().requires_grad_()
                    m = h * blk.mlp(q1).float()
                    p0 = (b - m.detach()).requires_grad_()
                    mp = blk.mlp_params()
                    g = torch.autograd.grad(m, [q1, *mp], gb)
                    gq = ga + g[0]
                    acc(mp, g[1:])
                    at = h * blk.attn(p0).float()
                    q0 = a - at.detach()
                    ap = blk.attn_params()
                    g = torch.autograd.grad(at, [p0, *ap], gq)
                    gp = gb + g[0]
                    acc(ap, g[1:])
                    a, b, ga, gb = q0, p0.detach(), gq, gp
                else:
                    # forward was (s, u) -> (u, v) with v = cs*s + cu*u + k f(u).  (a, b) = (u, v)
                    cs, cu, k = TWO_STEP[ctx.variant](h, ctx.alpha)
                    u = a.detach().requires_grad_()
                    kf = k * blk.f(u).float()
                    s = b - kf.detach()
                    if cu:
                        s = s - cu * a
                    s = s / cs
                    bp = list(blk.parameters())
                    g = torch.autograd.grad(kf, [u, *bp], gb)
                    acc(bp, g[1:])
                    gs, gu = cs * gb, ga + g[0]
                    if cu:
                        gu = gu + cu * gb
                    a, b, ga, gb = s, u.detach(), gs, gu
        params = [p for blk in ctx.blocks for p in blk.parameters()]
        return (ga, gb, None, None, None, None, *[grads.get(p) for p in params])


class GPT(nn.Module):
    def __init__(self, vocab=8192, d=384, n_layer=10, n_head=6, ctx=512, variant="standard", h=1.0, alpha=0.5):
        super().__init__()
        assert variant in VARIANTS, variant
        self.variant, self.h, self.alpha = variant, h, alpha
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("attn_out.weight") or n.endswith("mlp_out.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def stack(self, x):
        if self.variant == "standard":
            for blk in self.blocks:
                x = x + blk.attn(x)
                x = x + blk.mlp(x)
            return x
        if self.variant == "checkpoint":  # recomputation baseline: store each block's input only
            for blk in self.blocks:
                x = checkpoint(_std_block, blk, x, use_reentrant=False)
            return x
        params = [p for blk in self.blocks for p in blk.parameters()]
        a, b = RevStack.apply(x, x, self.variant, self.h, self.alpha, self.blocks, *params)
        return b

    def forward(self, idx, targets, loss_chunk=2048):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.ln_f(self.stack(x))
        # Chunked cross-entropy: only one chunk of logits (chunk x vocab) is alive at a time, in every variant.
        x, y = x.reshape(-1, x.size(-1)), targets.reshape(-1)
        total = 0.0
        for i in range(0, x.size(0), loss_chunk):
            total = total + checkpoint(_ce_sum, x[i:i + loss_chunk], y[i:i + loss_chunk], self.head.weight,
                                       use_reentrant=False)
        return total / y.numel()


def _std_block(blk, x):
    x = x + blk.attn(x)
    return x + blk.mlp(x)


def _ce_sum(x, y, w):
    return F.cross_entropy(F.linear(x, w).float(), y, reduction="sum")


@torch.no_grad()
def reconstruction_error(model, idx):
    """Run the reversible stack forward, then backwards via the inverse rule; report max |x0_rebuilt - x0|."""
    assert model.variant in ("euler", *TWO_STEP)
    B, T = idx.shape
    x = (model.tok(idx) + model.pos(torch.arange(T, device=idx.device))).float()
    with amp_ctx():
        a, b = x, x
        for blk in model.blocks:
            a, b = (_euler_step(blk, a, b, model.h) if model.variant == "euler"
                    else _two_step(blk, a, b, TWO_STEP[model.variant](model.h, model.alpha)))
        for blk in reversed(model.blocks):
            if model.variant == "euler":
                p0 = b - model.h * blk.mlp(a).float()
                a, b = a - model.h * blk.attn(p0).float(), p0
            else:
                cs, cu, k = TWO_STEP[model.variant](model.h, model.alpha)
                a, b = (b - cu * a - k * blk.f(a).float()) / cs, a
    return max((a - x).abs().max().item(), (b - x).abs().max().item()) / x.abs().max().item()


# ----------------------------------------------------------------------------- data

class Tokens:
    def __init__(self, split):
        self.data = np.memmap(os.path.join(DATA, f"{split}.bin"), dtype=np.uint16, mode="r")

    def batch(self, B, T, gen):
        ix = torch.randint(len(self.data) - T - 1, (B,), generator=gen).tolist()
        x = torch.from_numpy(np.stack([self.data[i:i + T + 1].astype(np.int64) for i in ix]))
        return x[:, :-1].pin_memory().cuda(non_blocking=True), x[:, 1:].pin_memory().cuda(non_blocking=True)


# ----------------------------------------------------------------------------- training

def gpu_state():
    """'temp C / SM MHz / mem MHz' from nvidia-smi; the laptop GPU throttles, so the logs record it."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        t, sm, mem = (int(v) for v in out.strip().split(","))
        return f"{t}C {sm}/{mem}MHz"
    except Exception:
        return "n/a"


DEFAULTS = dict(name="run", variant="standard", h=1.0, alpha=0.5, d=384, n_layer=10, n_head=6, ctx=512,
                batch=16, tokens=50_000_000, lr=1e-3, warmup_frac=0.02, min_lr_frac=0.1, wd=0.0, clip=1.0,
                eval_every=500, ckpt_every=None, eval_batches=20, final_eval_batches=100, log_every=50, seed=1337)


@torch.no_grad()
def evaluate(model, data, B, T, n, seed=0):
    model.eval()
    gen, tot = torch.Generator().manual_seed(seed), 0.0
    for _ in range(n):
        x, y = data.batch(B, T, gen)
        with amp_ctx():
            tot += model(x, y).item()
    model.train()
    return tot / n


def run_or_load(**kw):
    """Train, or if results/<name>.json already exists (an earlier execution of this notebook), replay its log."""
    name = kw.get("name", DEFAULTS["name"])
    jp, lp = (os.path.join(RESULTS, f"{name}.{e}") for e in ("json", "log"))
    if os.path.exists(jp) and os.path.exists(lp):
        print(f"(results/{name}.json exists: replaying the log of that run instead of retraining)")
        print(open(lp, encoding="utf-8").read(), end="")
        return json.load(open(jp))
    return train(**kw)


def train(**kw):
    cfg = {**DEFAULTS, **kw}
    os.makedirs(RESULTS, exist_ok=True)
    log_path = os.path.join(RESULTS, f"{cfg['name']}.log")
    ck_dir = os.path.join(HERE, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    resume_path = os.path.join(ck_dir, f"{cfg['name']}_resume.pt")
    resume = torch.load(resume_path, map_location="cpu", weights_only=False) if os.path.exists(resume_path) else None
    if resume is not None:
        # replay the log up to the checkpoint and drop anything written after it
        with open(log_path, "r+", encoding="utf-8") as fh:
            fh.truncate(resume["log_bytes"])
        print(open(log_path, encoding="utf-8").read(), end="", flush=True)
    logf = open(log_path, "a" if resume is not None else "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    torch.manual_seed(cfg["seed"])
    B, T = cfg["batch"], cfg["ctx"]
    tok_per_step = B * T
    steps = cfg["tokens"] // tok_per_step
    warm = max(1, int(cfg["warmup_frac"] * steps))
    model = GPT(d=cfg["d"], n_layer=cfg["n_layer"], n_head=cfg["n_head"], ctx=T, variant=cfg["variant"],
                h=cfg["h"], alpha=cfg["alpha"]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=cfg["wd"],
                            fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP and AMP_DTYPE is torch.float16)
    train_data, val_data = Tokens("train"), Tokens("val")
    gen = torch.Generator().manual_seed(cfg["seed"])

    def lr_at(s):
        if s < warm:
            return cfg["lr"] * (s + 1) / warm
        r = (s - warm) / max(1, steps - warm)
        return cfg["lr"] * (cfg["min_lr_frac"] + (1 - cfg["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * r)))

    ckpt_every = cfg["ckpt_every"] or cfg["eval_every"]
    hist = dict(step=[], loss=[], val_step=[], val_loss=[], tok_s=[])
    start, train_time, skipped, peak0, peak_res0 = 0, 0.0, 0, 0.0, 0.0
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        scaler.load_state_dict(resume["scaler"])
        gen.set_state(resume["gen"])
        start, train_time, skipped, hist = resume["step"], resume["train_time"], resume["skipped"], resume["hist"]
        peak0, peak_res0 = resume["peak"], resume["peak_res"]
        del resume
        log(f"[{cfg['name']}] RESUMED from checkpoint at step {start}/{steps} "
            f"({start * tok_per_step / 1e6:.1f}M tokens, {train_time / 60:.1f} min of training so far)")
    if start == 0:
        log(f"[{cfg['name']}] variant={cfg['variant']} h={cfg['h']} alpha={cfg['alpha']} params={model.n_params() / 1e6:.2f}M "
        f"batch={B}x{T}={tok_per_step} tokens/step steps={steps} tokens={steps * tok_per_step / 1e6:.1f}M "
        f"lr={cfg['lr']} precision={'fp32' if not AMP else str(AMP_DTYPE).split('.')[-1]} "
        f"{torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0, loss_acc, n_acc = time.time(), 0.0, 0
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = train_data.batch(B, T, gen)
        with amp_ctx():
            loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped += scaler.get_scale() < scale
        opt.zero_grad(set_to_none=True)
        loss_acc += loss.item()
        n_acc += 1
        if (step + 1) % cfg["log_every"] == 0 or step == steps - 1:
            torch.cuda.synchronize()
            dt = time.time() - t0
            train_time += dt
            tps = n_acc * tok_per_step / dt
            hist["step"].append(step + 1)
            hist["loss"].append(loss_acc / n_acc)
            hist["tok_s"].append(tps)
            msg = (f"step {step + 1:5d}/{steps} | loss {loss_acc / n_acc:.4f} | lr {lr_at(step):.2e} | "
                   f"{tps:,.0f} tok/s | peak {torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB | "
                   f"{(step + 1) * tok_per_step / 1e6:.1f}M tok | {train_time / 60:.1f} min | gpu {gpu_state()}")
            if (step + 1) % cfg["eval_every"] == 0 and step != steps - 1:
                vl = evaluate(model, val_data, 16, T, cfg["eval_batches"])
                hist["val_step"].append(step + 1)
                hist["val_loss"].append(vl)
                msg += f" | val {vl:.4f}"
            log(msg)
            if (step + 1) % ckpt_every == 0 and step != steps - 1:
                tmp = resume_path + ".tmp"
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                                gen=gen.get_state(), step=step + 1, train_time=train_time, skipped=skipped,
                                hist=hist, log_bytes=logf.tell(),
                                peak=max(peak0, torch.cuda.max_memory_allocated() / 2**20),
                                peak_res=max(peak_res0, torch.cuda.max_memory_reserved() / 2**20)), tmp)
                os.replace(tmp, resume_path)
            loss_acc, n_acc = 0.0, 0
            t0 = time.time()
    peak = max(peak0, torch.cuda.max_memory_allocated() / 2**20)
    peak_res = max(peak_res0, torch.cuda.max_memory_reserved() / 2**20)
    final_val = evaluate(model, val_data, 16, T, cfg["final_eval_batches"], seed=123)  # same 819k val tokens for every run
    hist["val_step"].append(steps)
    hist["val_loss"].append(final_val)
    res = dict(cfg=cfg, params=model.n_params(), steps=steps, tokens_seen=steps * tok_per_step,
               gpu=torch.cuda.get_device_name(0), precision=("fp32" if not AMP else str(AMP_DTYPE).split(".")[-1]),
               final_train_loss=float(np.mean(hist["loss"][-5:])), final_val_loss=final_val,
               tok_per_s=steps * tok_per_step / train_time, train_minutes=train_time / 60,
               peak_alloc_mib=peak, peak_reserved_mib=peak_res, amp_skipped_steps=int(skipped), hist=hist)
    if cfg["variant"] not in ("standard", "checkpoint"):
        x, _ = val_data.batch(4, T, torch.Generator().manual_seed(7))
        res["reconstruction_rel_err"] = reconstruction_error(model, x)
    log(f"[{cfg['name']}] DONE final train loss {res['final_train_loss']:.4f} | final val loss {final_val:.4f} | "
        f"{res['tok_per_s']:,.0f} tok/s | peak alloc {peak:,.0f} MiB (reserved {peak_res:,.0f}) | "
        f"{res['train_minutes']:.1f} min | AMP skipped steps {skipped}"
        + (f" | reconstruction rel err {res['reconstruction_rel_err']:.2e}" if "reconstruction_rel_err" in res else ""))
    logf.close()
    torch.save(model.state_dict(), os.path.join(ck_dir, f"{cfg['name']}.pt"))
    with open(os.path.join(RESULTS, f"{cfg['name']}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    if os.path.exists(resume_path):
        os.remove(resume_path)
    del model, opt
    torch.cuda.empty_cache()
    return res


@torch.no_grad()
def generate(model, tok, prompt, n=120, temperature=0.8, top_k=40, seed=0):
    torch.manual_seed(seed)
    model.eval()
    idx = torch.tensor([tok.encode(prompt).ids], device="cuda")
    for _ in range(n):
        x = idx[:, -model.pos.num_embeddings:]
        h = model.ln_f(model.stack(model.tok(x) + model.pos(torch.arange(x.size(1), device="cuda"))))
        logits = model.head(h[:, -1]) / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        if idx[0, -1].item() == tok.token_to_id("<|endoftext|>"):
            break
    return tok.decode(idx[0].tolist())


# ----------------------------------------------------------------------------- probes

def probe(variant, batch, steps=3, h=1.0, n_layer=10, ctx=512):
    """Peak memory (MiB) and tok/s for a few training steps, or None on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = opt = None
    try:
        model = GPT(variant=variant, h=h, n_layer=n_layer, ctx=ctx).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        scaler = torch.amp.GradScaler("cuda", enabled=AMP and AMP_DTYPE is torch.float16)
        gen = torch.Generator().manual_seed(0)
        data = Tokens("val")
        times = []
        for _ in range(steps):
            x, y = data.batch(batch, ctx, gen)
            torch.cuda.synchronize()
            t = time.time()
            with amp_ctx():
                loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.time() - t)
        peak = torch.cuda.max_memory_allocated() / 2**20
        state = model.n_params() * 16 / 2**20  # fp32 weight + grad + Adam m + v
        return dict(peak_mib=peak, state_mib=state, act_mib=peak - state,
                    tok_s=batch * ctx / np.median(times[1:] if len(times) > 1 else times))
    except torch.OutOfMemoryError:
        return None
    finally:
        del model, opt
        torch.cuda.empty_cache()


def max_batch(variant, start=8, limit=4096, **kw):
    """Largest batch that trains without OOM: doubling, then binary search."""
    lo, hi = 0, None
    b = start
    while b <= limit:
        if probe(variant, b, steps=2, **kw) is None:
            hi = b
            break
        lo, b = b, b * 2
    if hi is None:
        return lo
    while hi - lo > max(1, lo // 16):
        mid = (lo + hi) // 2
        if probe(variant, mid, steps=2, **kw) is None:
            hi = mid
        else:
            lo = mid
    return lo
