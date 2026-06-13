# Experiment Results

## Canonical Baseline vs Ablations

All runs use the same APTOS 2019 validation and held-out test splits and the
same green-channel preprocessing contract. The baseline uses an EfficientNetB0
softmax classifier with balanced class weights. The SMOTE run uses a
class-balanced synthetic training set and no class weights. The
ordinal-regression run keeps the same real-image training set and
class-balanced sample weighting, but replaces the softmax head with a single
severity-score output. Its class thresholds are tuned on the validation split
for QWK. The EfficientNetB1 and EfficientNetB3 runs keep the softmax head and
real-image class-weighted training recipe, but swap the backbone.

| Metric | B0 baseline | SMOTE | Ordinal regression | B1 backbone | B3 backbone |
| --- | ---: | ---: | ---: | ---: | ---: |
| QWK | 0.8655 | 0.8629 | 0.8037 | 0.8595 | 0.8908 |
| Accuracy | 0.7842 | 0.7869 | 0.6612 | 0.7678 | 0.8005 |
| Macro sensitivity | 0.6431 | 0.5997 | 0.4520 | 0.6154 | 0.6402 |
| Macro specificity | 0.9492 | 0.9483 | 0.9165 | 0.9449 | 0.9530 |
| Macro ROC-AUC | 0.9204 | 0.9083 | 0.8032 | 0.9265 | 0.9320 |

| Grade | B0 sensitivity | SMOTE sensitivity | Ordinal-regression sensitivity | B1 sensitivity | B3 sensitivity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 No DR | 0.9598 | 0.9648 | 0.8392 | 0.9548 | 0.9648 |
| 1 Mild | 0.5333 | 0.2667 | 0.3333 | 0.5333 | 0.6333 |
| 2 Moderate | 0.6207 | 0.7241 | 0.6437 | 0.5747 | 0.6437 |
| 3 Severe | 0.6471 | 0.5882 | 0.3529 | 0.5294 | 0.3529 |
| 4 Proliferative | 0.4545 | 0.4545 | 0.0909 | 0.4848 | 0.6061 |

## Interpretation

SMOTE does not improve the overall model for the current pipeline. Its QWK is
nearly tied with the baseline and its accuracy is slightly higher, but macro
sensitivity drops because grade 1 recall falls from 0.5333 to 0.2667. Grade 2
recall improves, while grade 4 recall is unchanged.

The ordinal-regression run also does not improve the current model. Although it
matches the proposal's motivation that DR grades are ordered, the held-out test
QWK drops to 0.8037 and macro sensitivity drops to 0.4520. The validation-tuned
thresholds were approximately `[0.92, 1.84, 3.13, 4.24]`, with validation
threshold QWK 0.8359, but those thresholds generalize poorly to the held-out
test split, especially for grade 4 sensitivity.

EfficientNetB1 is closer to the selected baseline and reached a higher best
validation QWK during training, but its held-out test QWK is 0.8595 versus
0.8655 for B0. It improves macro ROC-AUC and slightly improves grade 4
sensitivity, but loses macro sensitivity overall because grade 2 and grade 3
recall drop.

EfficientNetB3 is the strongest run on the primary metric. It improves held-out
test QWK from 0.8655 to 0.8908, accuracy from 0.7842 to 0.8005, macro
specificity from 0.9492 to 0.9530, and macro ROC-AUC from 0.9204 to 0.9320.
It also improves grade 1 and grade 4 sensitivity. The main trade-off is grade
3 sensitivity, which falls from 0.6471 to 0.3529. Because QWK is the primary
metric, EfficientNetB3 is the current best candidate, but the severe-class
recall drop must be highlighted in the paper discussion.

The selected report result should therefore move from the class-weighted
EfficientNetB0 baseline to the class-weighted EfficientNetB3 backbone run, with
the severe-grade sensitivity limitation reported explicitly.

## Explainability Comparison

The selected EfficientNetB3 model now has qualitative outputs for three XAI
methods:

| Method | Artifact pattern | Notes |
| --- | --- | --- |
| Grad-CAM | `artifacts/b3/figures/gradcam_*.png` | Original XAI method; useful but sometimes attends to optic disc or border regions. |
| Grad-CAM++ | `artifacts/b3/figures/gradcampp_*.png` | Uses stronger spatial weighting and often produces sharper heatmaps than Grad-CAM. |
| Score-CAM top-k | `artifacts/b3/figures/scorecam_*.png` | Computed on a limited sample set using the top 32 activation channels per image to avoid full masked forward passes per image. Outputs are broader and more diffuse. |

These artifacts support the proposal's qualitative XAI comparison. They do not
constitute lesion-mask validation: no IDRiD masks, pointing-game accuracy, or
IoU scores are used in the current scope.
