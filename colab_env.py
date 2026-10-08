"""Google Colab support for the Session 13 runs: Drive persistence, and picking the precision.

A Colab runtime is deleted the moment it disconnects, and a free session disconnects long before
all the runs in `reversible_llm_colab.ipynb` have finished. So everything that must survive a
disconnect lives in Google Drive, and everything the training loop touches on the hot path lives
on the runtime's local disk:

    local disk (fast, thrown away)         Google Drive (slow, survives)
    <repo>/data/{train,val}.bin            era-v5-s13/data/          copied down once per session
    <repo>/data/tokenizer.json             era-v5-s13/data/
    <repo>/results/*.{log,json,csv,png}    era-v5-s13/results/       mirrored every SYNC_SECONDS
    <repo>/checkpoints/*.pt                era-v5-s13/checkpoints/   mirrored every SYNC_SECONDS

Pointing `np.memmap` at the Drive FUSE mount would turn every training batch into a network read,
and `torch.save` straight to Drive would stall the loop for a minute every few hundred steps, so
neither is pointed at Drive. A background thread copies the changed files up instead, which is why
losing the runtime costs at most SYNC_SECONDS plus one checkpoint interval of training.

Nothing here is Colab-specific except `mount()`; on any other machine with a persistent disk, set
DRIVE to a directory on it and the same restore/sync logic applies.
"""
import os
import shutil
import threading
import time

DRIVE = "/content/drive/MyDrive/era-v5-s13"
HERE = os.path.dirname(os.path.abspath(__file__))
# (name on local disk, name under DRIVE). Only top-level files are mirrored, which is what keeps
# the 249 MB of downloaded parquet in data/raw/ out of Drive.
DIRS = [("data", "data"), ("results", "results"), ("checkpoints", "checkpoints")]
SYNC_SECONDS = 120

_thread = None
stats = dict(syncs=0, files=0, last=None, last_error=None)


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- mount

def mount(drive=None):
    """Mount Drive (no-op outside Colab or if already mounted) and create the folders."""
    global DRIVE
    if drive:
        DRIVE = drive
    if not os.path.ismount("/content/drive") and DRIVE.startswith("/content/drive"):
        from google.colab import drive as gdrive  # only exists on Colab
        gdrive.mount("/content/drive")
    for _, d in DIRS:
        os.makedirs(os.path.join(DRIVE, d), exist_ok=True)
    log(f"[drive] {DRIVE}")
    return DRIVE


# ----------------------------------------------------------------------------- copying

def _mirror(src_dir, dst_dir):
    """Copy top-level files whose size or mtime differs. Writes via .part + rename so a copy that
    is cut off mid-flight never leaves a plausible-looking truncated checkpoint behind."""
    copied = []
    names = sorted(os.listdir(src_dir)) if os.path.isdir(src_dir) else []
    for name in names:
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src) or name.endswith((".part", ".tmp")):
            continue
        dst = os.path.join(dst_dir, name)
        s = os.stat(src)
        if os.path.exists(dst):
            d = os.stat(dst)
            if d.st_size == s.st_size and d.st_mtime >= s.st_mtime - 1:
                continue
        os.makedirs(dst_dir, exist_ok=True)
        part = dst + ".part"
        shutil.copy2(src, part)
        try:
            os.replace(part, dst)
        except OSError:  # a FUSE mount that refuses rename
            shutil.copy2(src, dst)
            os.remove(part)
        copied.append(name)
    return copied


def sync_now(quiet=False):
    """Copy local results/checkpoints/data up to Drive."""
    copied = []
    for local, remote in DIRS:
        copied += _mirror(os.path.join(HERE, local), os.path.join(DRIVE, remote))
    stats["syncs"] += 1
    stats["files"] += len(copied)
    stats["last"] = time.time()
    if copied and not quiet:
        log(f"[drive] uploaded {len(copied)} file(s): {', '.join(copied[:6])}"
            + (f" (+{len(copied) - 6} more)" if len(copied) > 6 else ""))
    return copied


def restore(verify=True):
    """Copy anything already in Drive down to local disk, so a fresh runtime continues where the
    last one stopped. Finished runs are then replayed from their logs by `revlm.run_or_load` and
    interrupted ones resume from `checkpoints/<run>_resume.pt`."""
    got = {}
    for local, remote in DIRS:
        os.makedirs(os.path.join(HERE, local), exist_ok=True)
        got[local] = _mirror(os.path.join(DRIVE, remote), os.path.join(HERE, local))
    for local, files in got.items():
        log(f"[drive] restored {len(files):2d} file(s) into {local}/" + (f": {', '.join(files)}" if files else ""))
    if verify:
        drop_broken_checkpoints()
    done = sorted(f[:-5] for f in os.listdir(os.path.join(HERE, "results"))
                  if f.endswith(".json") and f != "max_batch.json")
    part = sorted(f[:-len("_resume.pt")] for f in os.listdir(os.path.join(HERE, "checkpoints"))
                  if f.endswith("_resume.pt"))
    log(f"[drive] runs already finished: {', '.join(done) or 'none'}")
    log(f"[drive] runs to resume mid-way: {', '.join(part) or 'none'}")
    return got


def drop_broken_checkpoints():
    """A resume checkpoint that cannot be loaded (a copy interrupted by an earlier disconnect)
    would crash the run that reads it, so delete it here and let that run start over instead."""
    try:
        import torch
    except ImportError:
        return
    ck = os.path.join(HERE, "checkpoints")
    for name in sorted(os.listdir(ck)) if os.path.isdir(ck) else []:
        if not name.endswith("_resume.pt"):
            continue
        p = os.path.join(ck, name)
        try:
            torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            log(f"[drive] {name} is unreadable ({type(e).__name__}), deleting it: that run restarts from step 0")
            os.remove(p)
            for d in (ck, os.path.join(DRIVE, "checkpoints")):
                if os.path.exists(os.path.join(d, name)):
                    os.remove(os.path.join(d, name))


# ----------------------------------------------------------------------------- background sync

def start_sync(interval=None):
    """Mirror local -> Drive every `interval` seconds, quietly, for as long as the runtime lives."""
    global _thread
    if _thread and _thread.is_alive():
        log(f"[drive] background sync already running (every {SYNC_SECONDS}s)")
        return _thread

    def loop():
        while True:
            time.sleep(interval or SYNC_SECONDS)
            try:
                sync_now(quiet=True)
            except Exception as e:  # a transient Drive error must not kill the thread
                stats["last_error"] = f"{type(e).__name__}: {e}"

    _thread = threading.Thread(target=loop, daemon=True, name="drive-sync")
    _thread.start()
    log(f"[drive] background sync started: results/, checkpoints/ and data/ go up every "
        f"{interval or SYNC_SECONDS}s, so a disconnect costs at most that much training")
    return _thread


def sync_report():
    age = "never" if not stats["last"] else f"{time.time() - stats['last']:.0f}s ago"
    log(f"[drive] {stats['syncs']} syncs, {stats['files']} files uploaded, last {age}"
        + (f", last error {stats['last_error']}" if stats["last_error"] else ""))


# ----------------------------------------------------------------------------- precision

def matmul_tflops(dtype, n=4096, iters=20):
    import torch
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = time.time() - t
    del a, b
    torch.cuda.empty_cache()
    return iters * 2 * n**3 / dt / 1e12


def time_step(variant, dtype, batch=8, **kw):
    """Measured tok/s of real training steps for one variant at one precision (None on OOM)."""
    import torch

    import revlm
    amp, amp_dtype = revlm.AMP, getattr(revlm, "AMP_DTYPE", torch.float16)
    revlm.AMP = dtype is not None
    revlm.AMP_DTYPE = dtype or torch.float16
    try:
        return revlm.probe(variant, batch, steps=6, **kw)
    finally:
        revlm.AMP, revlm.AMP_DTYPE = amp, amp_dtype


def choose_precision(variants=("standard", "euler"), batch=8, min_speedup=1.15):
    """Measure fp32 against the mixed-precision options on *this* GPU with real training steps and
    pick the fastest, requiring a clear win before leaving fp32 behind.

    The laptop runs chose fp32 because the GTX 1660 Ti has no tensor cores and its fp16 matmuls
    measured 9x slower than fp32. Every Colab GPU does have tensor cores, so the answer here is
    normally the opposite -- but it is measured, not assumed, because the choice has to hold for
    the reversible stack too, where each block runs twice per step.

    Sets revlm.AMP / revlm.AMP_DTYPE and returns the table it measured.
    """
    import torch

    import revlm
    opts = [("fp32", None), ("fp16", torch.float16)]
    if torch.cuda.is_bf16_supported():
        opts.append(("bf16", torch.bfloat16))
    rows = {}
    for label, dtype in opts:
        tf = matmul_tflops(dtype or torch.float32)
        speeds = {}
        for v in variants:
            r = time_step(v, dtype, batch=batch, h=1.0)
            speeds[v] = r["tok_s"] if r else float("nan")
        rows[label] = dict(dtype=dtype, matmul_tflops=tf, **speeds)
        log(f"{label:5s} 4096^3 matmul {tf:6.2f} TFLOPS | "
            + "  ".join(f"{v} {speeds[v]:,.0f} tok/s" for v in variants))
    base = min(rows["fp32"][v] for v in variants)
    best, gain = "fp32", 1.0
    for label in rows:
        if label == "fp32":
            continue
        g = min(rows[label][v] for v in variants) / base
        if g > gain:
            best, gain = label, g
    if gain < min_speedup:
        best = "fp32"
    revlm.AMP = rows[best]["dtype"] is not None
    revlm.AMP_DTYPE = rows[best]["dtype"] or torch.float16
    log(f"\n-> training in {best}" + ("" if best == "fp32" else f" ({gain:.2f}x the fp32 throughput on the slowest variant)")
        + f"  [revlm.AMP={revlm.AMP}, dtype={revlm.AMP_DTYPE}]")
    return best, rows
