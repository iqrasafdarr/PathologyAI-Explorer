# CLAM: Architecture Exploration

This document explains the architecture of **CLAM (Clustering-constrained
Attention Multiple Instance Learning)**, the leading open-source framework
for attention-based, weakly-supervised whole-slide image classification.
**This is an architectural walkthrough, not a reproduction.** No CLAM model
is trained or evaluated in this repository — see "What Was and Wasn't Done"
below for the exact, honest scope.

## Why CLAM

CLAM (Lu et al., *Nature Biomedical Engineering*, 2021) was designed to
solve the problem introduced in `docs/wsi_and_mil.md`: given a WSI and only
a slide-level label (no patch-level annotations), how do you train a model
that both classifies the slide **and** identifies which regions drove that
decision? It's one of the most widely adopted MIL frameworks in
computational pathology precisely because it addresses both needs at once,
with a design that scales to large datasets and stays interpretable.

## Architecture Overview

CLAM extends the basic attention-MIL idea (described in
`docs/wsi_and_mil.md`) with two additional design choices:

### 1. Attention-Based Pooling (shared with standard attention-MIL)
Each patch is embedded via a CNN feature extractor, then a learned
attention network scores each patch embedding's relevance. Patch
embeddings are combined into a slide-level representation via a weighted
sum, where the weights are the learned attention scores — this is the same
mechanism described in `docs/wsi_and_mil.md`, and it's what gives CLAM its
interpretability: the attention scores double as a heatmap over the slide,
highlighting which regions the model considered most diagnostic.

### 2. Clustering-Constrained Instance-Level Auxiliary Loss
This is CLAM's key addition beyond plain attention-MIL. In parallel with
the slide-level classification task, CLAM trains an **auxiliary
clustering objective** on the patches with the *highest* and *lowest*
attention scores within each slide:
- The highest-attention patches are pushed toward being classified
  consistently with the slide-level label (they're the model's strongest
  candidates for containing the diagnostic evidence).
- The lowest-attention patches are pushed toward the "normal tissue"
  class.

This auxiliary loss acts as a regularizer: it encourages the attention
mechanism to genuinely separate diagnostically relevant patches from
irrelevant ones, rather than learning a diffuse or noisy attention
distribution that happens to work for the slide-level label alone. In
CLAM's terminology this is called **instance-level clustering**, and it's
what "clustering-constrained" in the name refers to.

### 3. Multi-Class Extension (CLAM-MB)
The original single-attention-branch version (CLAM-SB) uses one attention
network shared across classes. For multi-class problems, CLAM-MB
(multi-branch) learns a separate attention network per class, so each
class can attend to different regions of the same slide — relevant for
subtyping tasks with more than two categories. PCam and the scope of this
repository are binary (tumor / no tumor), so CLAM-SB is the architecturally
relevant variant here.

## Relationship to This Repository's Own Work

| | This repo's Module A1 (ResNet50/PCam) | CLAM |
|---|---|---|
| Input | Pre-extracted, individually-labeled 96×96 patches | Whole slides, slide-level label only |
| Supervision | Fully supervised at the patch level | Weakly supervised (MIL) — no patch labels |
| Output | Per-patch classification | Per-slide classification + attention heatmap |
| Feature extraction | ResNet50, fine-tuned end-to-end | Typically a frozen or lightly-tuned feature extractor, followed by attention-MIL layers |

These are genuinely different tasks operating on different label
granularity — Module A1 could plausibly serve as (part of) the feature
extraction stage feeding into a CLAM-style pipeline, but that integration
is not implemented here.

## What Was and Wasn't Done

**Attempted:** installing the official CLAM repository
([github.com/mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM)) and
reviewing its documented workflow for running its provided demo/tutorial
data.

**Not reproduced:** a full CLAM training run. This was scoped out
deliberately, for reasons consistent with the honest-scope principle
followed throughout this repository:

1. **Data requirement mismatch.** CLAM's pipeline expects whole-slide image
   files (`.svs`/`.tiff`) as input to its own patching and feature-extraction
   scripts — it is not designed to consume PCam's already-extracted
   96×96 patches directly. Running CLAM as intended would require
   downloading raw CAMELYON16 WSIs, which — as discussed in the original
   project scoping for this repository — is computationally unrealistic
   given available time and storage (CAMELYON16 alone is several hundred
   GB).
2. **Dependency footprint.** CLAM's environment (OpenSlide, specific
   `torch`/`torchvision` version pins, its own feature-extraction
   pipeline) is substantial enough that isolating it from this
   repository's main environment would be its own undertaking, separate
   from the actual research question.

Rather than force a partial or misleading run, this is documented
explicitly as **future work**: a natural next step would be to obtain a
small number of CAMELYON16 WSIs (or another WSI-scale dataset with
compatible tooling), run CLAM's own patching/feature-extraction scripts on
them, and train CLAM-SB on that small set — reporting whatever results
that produces, honestly, the same way Module A1's results are reported.

## Summary

CLAM's core contribution — clustering-constrained attention-MIL — is
explained here at the architecture level, and is understood well enough to
motivate why it's a stronger design than naive patch-vote aggregation. Its
actual training was not attempted in this repository, for the specific,
disclosed reasons above, rather than left unexplained or silently skipped.
