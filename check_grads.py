"""Check that the reversible backward pass gives the same gradients as plain autograd on the same update rule."""
import torch

from revlm import GPT, TWO_STEP, _euler_step, _two_step


def autograd_stack(model, x):
    a, b = x, x
    for blk in model.blocks:
        a, b = (_euler_step(blk, a, b, model.h) if model.variant == "euler"
                else _two_step(blk, a, b, TWO_STEP[model.variant](model.h, model.alpha)))
    return b


def main():
    torch.manual_seed(0)
    idx = torch.randint(0, 8192, (2, 64), device="cuda")
    tgt = torch.randint(0, 8192, (2, 64), device="cuda")
    for variant, h in [("midpoint", 0.5), ("leapfrog", 0.5), ("blend", 1.0), ("euler", 1.0)]:
        model = GPT(variant=variant, h=h, n_layer=4, ctx=64).cuda().double()
        model(idx, tgt).backward()
        g_rev = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        x = model.tok(idx) + model.pos(torch.arange(64, device="cuda"))
        loss = torch.nn.functional.cross_entropy(model.head(model.ln_f(autograd_stack(model, x))).view(-1, 8192),
                                                 tgt.view(-1))
        loss.backward()
        g_ref = [p.grad for p in model.parameters()]
        err = max(((a - b).abs().max() / (b.abs().max() + 1e-30)).item() for a, b in zip(g_rev, g_ref))
        print(f"{variant:9s} max relative grad difference vs autograd: {err:.2e}", flush=True)


if __name__ == "__main__":
    main()
