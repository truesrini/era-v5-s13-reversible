# Reversible transformers: a 20M GPT trained for 50M tokens

ERA V5, Session 13 assignment. A 21M-parameter GPT trained on TinyStories for 50M tokens with a standard
residual stack, then with a reversible stack at the same batch, then with a reversible stack at the largest
batch that fits. All numbers below come from one run of `reversible_llm_colab.ipynb` on a **Colab Tesla T4**,
fp32.

| | |
|---|---|
| model | GPT, 10 layers, width 384, 6 heads, context 512, tied embeddings, 21.05M parameters |
| data | TinyStories, byte-level BPE tokenizer (vocab 8,192) trained here |
| optimizer | AdamW (0.9, 0.95), peak lr 1e-3, 2% warmup, cosine to 10%, grad clip 1.0, no dropout or weight decay |
| code | `revlm.py` (model, reversible backward pass, training), `prep_data.py` (data), `check_grads.py` (gradient check) |

## Results

| run | variant | batch | tokens | steps | final train loss | final val loss | tokens/s | peak alloc (MiB) | train time (min) | reconstruction rel. err. |
|---|---|---|---|---|---|---|---|---|---|---|
| standard_b16 | standard | 16 | 50.0M | 6,103 | 1.7879 | **1.7834** | **18,423** | 2,425 | 45.2 | - |
| euler_b16 | Euler | 16 | 50.0M | 6,103 | 1.9281 | 1.9210 | 14,016 | **649** | 59.5 | 1.8e-03 |
| rev_maxbatch | midpoint | 518 | 49.9M | 188 | 4.6213 | 4.5968 | 13,635 | 12,394 | 60.9 | **3.2e-01** |
| euler_b16_5M (screen) | Euler | 16 | 5.0M | 610 | 3.3665 | 3.2941 | 14,049 | 649 | 5.9 | 6.4e-04 |
| midpoint_b16_5M (screen) | midpoint | 16 | 5.0M | 610 | 3.3534 | 3.2730 | 13,901 | 709 | 6.0 | 1.6e-03 |

Peak memory is `torch.cuda.max_memory_allocated` over the whole run, including the fp32 weights, gradients and
Adam state (about 320 MiB for this model). Reconstruction error is the relative error when the trained
reversible stack rebuilds its input from its output, which is what the backward pass relies on.

## Findings

**1. The baseline is reproducible across machines.** The same run on the laptop's GTX 1660 Ti (fp32, same seed)
ended at val loss 1.783 with a peak of 2,423 MiB; on the T4 it ended at 1.7834 and 2,425 MiB. Only the speed
differs: 9,063 tok/s on the laptop, 18,423 on the T4.

**2. Reversibility works and cuts memory by 3.7x, for a modest cost.** The Euler (symplectic, two-stream) stack at
the same batch of 16 peaked at 649 MiB against 2,425 MiB, because it stores no per-layer activations: the backward
pass rebuilds each layer's input from its output. That rebuilding means every block runs twice per step, so it
trained at 76% of the baseline's speed (14,016 against 18,423 tok/s). Its val loss is 0.14 higher (1.921 against
1.783): the two-stream update is a different architecture from a residual stack, not a free re-implementation of
it.

**3. Which variant worked: Euler.** On equal 5M-token screens at batch 16, midpoint and Euler were effectively tied
(val loss 3.273 against 3.294), and both rebuilt their inputs to about 1e-3. The difference showed up under a large
batch and learning rate. With midpoint at batch 518 and lr 3e-3, the reconstruction error reached 0.32: the stack
was no longer numerically invertible, so the backward pass rebuilt the wrong activations and computed the wrong
gradients. The likely cause is that midpoint's inverse, `p[l-1] = p[l+1] - 2h f(p[l])`, amplifies rounding error from layer
to layer once the blocks' outputs grow, while Euler stayed at 1.8e-3 over a full 50M-token run. So Euler is the variant that worked
reliably.

**4. The maximum batch bought memory headroom, not speed or quality.** The probed maximum for the reversible stack on
the 15 GB T4 was 576 sequences (518 used, keeping 10% for fragmentation), about 32x the batch-16 run's tokens per
step, at a peak of 12.4 GB. But throughput did not rise (13,635 tok/s), because the T4 is already compute-bound at
batch 16. And with the token budget fixed at 50M, batch 518 leaves only 188 optimizer steps, 32x fewer than at batch
16, which on its own would leave the loss far higher. Together with midpoint's broken gradients, that is why this run
ended at val loss 4.60. A rerun of the max-batch case with Euler was started but not finished in time, so its
numbers are not reported here.

**5. Precision.** The notebook times fp32 against fp16/bf16 on the GPU it gets and keeps mixed precision only when it
is clearly faster. On this T4 it kept fp32, so every run above is fp32, the same as on the laptop.

## Running it

`reversible_llm_colab.ipynb` runs everything on a Colab or Kaggle GPU, keeping results and checkpoints in Google
Drive (or Kaggle's output) so a disconnect only costs a few minutes; see [COLAB.md](COLAB.md). `reversible_llm.ipynb`
is the original laptop notebook (GTX 1660 Ti, 6 GB), driven by `run_notebook.py`.
