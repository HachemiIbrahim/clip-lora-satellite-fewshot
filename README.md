# Few-shot CLIP adaptation for satellite scene classification, and a negative result on diffusion augmentation

## Summary

1. **Project 1 – efficient adaptation.** Zero-shot CLIP (ViT-B/32) is weak on satellite imagery (37.6%). LLM-generated class descriptions improve it modestly (+3.9 points). LoRA adaptation of the vision encoder with only 4–16 labeled images per class reaches 77–93%.
2. **Project 2 – generative augmentation (negative result).** I implemented a class-conditional DDPM from scratch (no `diffusers`) and tested whether its synthetic images improve the Project 1 classifier. They did not: accuracy dropped by about 4–12 points at every setting. A fidelity analysis shows why: the generator's images are recognized as their intended class only 36–51% of the time, versus 92% for real images.

## Project 1: CLIP + LoRA

**Setup.** CLIP ViT-B/32 (frozen). Zero-shot: cosine similarity between image embeddings and class text embeddings. LoRA: rank 8 on the query/value projections of the vision transformer, plus a linear classification head (~300k trainable parameters), 15 epochs, k ∈ {4, 8, 16} images per class.

| Method | Test accuracy (%) |
|---|---|
| Zero-shot, single template ("a satellite photo of …") | 37.61 |
| Zero-shot, LLM-generated description ensemble | 41.54 |
| LoRA few-shot, k=4 (single run, seed 42) | 77.04 |
| LoRA few-shot, k=8 (single run, seed 42) | 88.63 |
| LoRA few-shot, k=16 (single run, seed 42) | 91.19 |

**Variance check.** Re-running the same recipe over 3 different seeds (different k-shot draws) gave **80.7 ± 3.2** (k=4), **87.8 ± 2.1** (k=8), **92.7 ± 1.1** (k=16). The 3-seed numbers are the more reliable ones to quote; the single-run numbers above are consistent with them within noise.

**Cross-domain test.** The k=16 model, trained only on EuroSAT, was evaluated without retraining on 500 images (100 per class) from 5 semantically matching RESISC45 classes (forest, river, freeway→Highway, industrial→Industrial, dense residential→Residential): **64.6%**. *Caveat: I have not yet computed a reference on the same subset (e.g. zero-shot CLIP), so this number alone does not show how much is lost to domain shift.*

**Caveats.** The description-ensemble gain comes from one deterministic evaluation of one prompt set (no variance estimate). EuroSAT and RESISC45 differ in resolution and sensor, which this test does not separate.

## Project 2: diffusion augmentation

**Model.** Class-conditional DDPM written from scratch: 9.3M-parameter U-Net (residual blocks, self-attention at 16×16 and 8×8, sinusoidal timestep embedding), cosine noise schedule (T=1000), classifier-free guidance (10% label dropout), EMA weights (0.999), 10,000 training steps at batch size 64, DDIM and DDPM samplers.

**Generator quality (class fidelity = accuracy of a CLIP linear probe trained on real images, applied to synthetic images):**

| Source | Class fidelity (%) | Diversity |
|---|---|---|
| Real EuroSAT images | 92.1 | 0.104 |
| DDIM-100, guidance 1.5 (used for augmentation) | 36.0 | 0.110 |
| DDPM-1000, guidance 1.5 (best, ~10× slower) | 51.0 | 0.128 |

**Augmentation result (3 seeds, mean ± std test accuracy %, m = synthetic images added per class):**

| k real/class | m=0 | m=50 | m=100 | m=200 |
|---|---|---|---|---|
| 4 | **80.7 ± 3.2** | 72.2 ± 5.1 | 76.4 ± 2.5 | 76.4 ± 4.4 |
| 8 | **87.8 ± 2.1** | 77.5 ± 4.9 | 77.3 ± 3.6 | 78.9 ± 3.5 |
| 16 | **92.7 ± 1.1** | 81.0 ± 4.0 | 81.6 ± 5.2 | 81.3 ± 0.8 |

**Interpretation (what this does and does not show).** With this generator, augmentation hurts. Low class fidelity means a large share of synthetic images are effectively mislabeled, and at small k they outnumber the real images. This does **not** show that generated images cannot help CLIP: the generator is small, trained for few steps, and well below the real-data fidelity ceiling, and I only tested naive mixing (no filtering, no down-weighting, no synthetic-pretrain-then-real-finetune). Stronger generators are a separate question I did not test.

**Debugging notes (kept because they changed the conclusions).**
- An early run trained on unscaled 13-band reflectance data instead of RGB in [-1, 1]; the data-range check now asserts this.
- Unclipped DDIM sampling produced saturated noise even though training loss looked fine; clipping the x0 estimate fixed it.
- A first single-seed sweep suggested a gain at k=4; with 3 seeds and a better sampler the gain disappeared (it came from an unusually low baseline draw). Reported numbers are the 3-seed ones.

## Limitations and next steps
- Cross-domain test needs a reference baseline on the same RESISC45 subset.
- Test whether confidence-filtered or down-weighted synthetic data changes the sign of the result.
- Train the generator much longer (or use a stronger pretrained one) and re-measure fidelity before re-testing augmentation.


