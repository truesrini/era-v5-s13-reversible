# Running these experiments on Google Colab

`reversible_llm.ipynb` is the laptop notebook (GTX 1660 Ti, 6 GB, fp32). `reversible_llm_colab.ipynb` is the same
run plan on a Colab GPU, with the state kept in Google Drive so that a disconnected runtime costs minutes rather
than a whole run.

## Steps

1. Runtime > Change runtime type > **GPU** (T4 is enough; L4 or A100 is roughly 2-4x faster).
2. Open the notebook:
   [reversible_llm_colab.ipynb in Colab](https://colab.research.google.com/github/truesrini/era-v5-s13-reversible/blob/main/reversible_llm_colab.ipynb)
3. Run the first cell and allow Google Drive access. It clones this repo onto the runtime, creates
   `MyDrive/era-v5-s13/`, copies down anything earlier sessions left there, and starts a background upload.
4. Runtime > Run all, and leave the tab open. Section 5 prints an estimate of how long the remaining runs will take
   on the GPU you were given.

## When Colab disconnects

Reopen the notebook and run every cell again. Nothing is redone that was already done:

- a run with `results/<run>.json` in Drive **replays its saved log** instead of retraining;
- a run with `checkpoints/<run>_resume.pt` **continues from that step**, keeping its loss history, token count,
  optimizer state and data-order RNG;
- `data/train.bin`, `data/val.bin` and `data/tokenizer.json` come back from Drive, so the download and tokenizer
  training only ever happen once.

Checkpoints are written every 500 steps and uploaded within 2 minutes, so a lost runtime costs at most a few
minutes of training. You can also follow `MyDrive/era-v5-s13/results/<run>.log` from a phone while the tab is closed.

## What differs from the laptop notebook

| | laptop | Colab |
|---|---|---|
| precision | fp32 (no tensor cores on a 1660 Ti; fp16 matmuls measured 9x slower) | measured in section 2 across fp32/fp16/bf16 and the fastest is used, provided the reversible stack still reconstructs its inputs to better than 1e-3 |
| allocator cap | capped at free dedicated VRAM, because Windows WDDM silently spills into system RAM instead of raising OOM | not needed; Linux raises a real OOM |
| throughput notes | logs record GPU temperature and clocks, which drop under thermal throttling | datacentre-cooled, so the probe numbers match the long runs |
| persistence | local disk | `results/`, `checkpoints/` and `data/` mirrored to Drive by `colab_env.py` |
| max batch | bounded by 6 GB | bounded by 15 GB (T4) or 40 GB (A100), so the reversible max-batch run is much larger |

`revlm.py`, `prep_data.py` and `check_grads.py` are shared by both notebooks. The only change made for Colab was
`revlm.AMP_DTYPE`, so the precision can be chosen per machine rather than hard-coded.

## Carrying the laptop's runs over

The Colab notebook trains every run from scratch by default, so that loss, tokens/s and peak memory all come from
one GPU and stay comparable with each other.

If you would rather continue a run the laptop started, copy its files into `MyDrive/era-v5-s13/` before step 4:

- `results/<run>.log` and `results/<run>.json` for a run that **finished** on the laptop (it will be replayed, not
  retrained);
- `results/<run>.log` and `checkpoints/<run>_resume.pt` for a run that was **interrupted** (it resumes from there).

Two caveats. A resumed run's reported tokens/s and training minutes are the average over both machines, and its
peak memory is the larger of the two, so those two columns stop describing either GPU. And a run resumed into a
different precision changes mid-flight. The summary table prints a warning when it finds runs from more than one
GPU. For the assignment's headline numbers, fresh Colab runs are the cleaner comparison; the laptop's completed
baseline (val loss 1.783, 9,063 tok/s, peak 2,423 MiB) is worth reporting alongside as the 6 GB result.
