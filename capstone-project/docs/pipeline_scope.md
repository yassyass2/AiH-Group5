# Canonical DR Pipeline Scope

## Current Goal

The current goal is to fix and stabilize the technical pipeline before final paper results are reported.

## Included In Pipeline v1

- APTOS 2019 dataset only.
- Five-class diabetic retinopathy severity classification.
- Green-channel preprocessing:
  - black-border crop
  - aspect-ratio-preserving resize with zero padding
  - green-channel extraction
  - CLAHE with `clipLimit=2.0` and `tileGridSize=(8, 8)`
  - median blur with kernel size `3`
  - normalization to `float32 [0, 1]`
- EfficientNetB0 baseline.
- Class-weighted baseline training.
- Optional SMOTE experiment after baseline.
- Optional ordinal-regression head experiment with validation-tuned QWK thresholds.
- Optional EfficientNetB1/B3 backbone ablations.
- Qualitative XAI comparison:
  - Grad-CAM
  - Grad-CAM++
  - top-k Score-CAM for a limited sample set
- QWK, accuracy, per-class sensitivity/specificity, macro AUC, confusion matrix.

## Excluded Until Implemented

- EyePACS.
- Messidor-2.
- IDRiD lesion masks.
- Pointing-game accuracy.
- IoU against lesion masks.

## Data Contract

Saved preprocessed arrays use shape `(N, 224, 224, 1)`, dtype `float32`, and values in `[0, 1]`.

## Model Contract

The canonical model accepts one-channel green images in `[0, 1]`. The model graph is responsible for converting those tensors to the range and channel count expected by EfficientNet.

The selected model head remains the five-class softmax classifier unless an ablation demonstrably improves held-out test performance. The ordinal-regression head is implemented as an experiment, but its current held-out test result is lower than the class-weighted baseline.

The selected backbone remains EfficientNetB0 unless a larger-backbone ablation improves held-out test performance. EfficientNetB1 is implemented and trained as an ablation, but its current held-out test QWK is slightly lower than the B0 baseline.

## Paper Rule

The paper may only claim methods and datasets implemented in this repository. Planned extensions must be described as future work, not completed work.
