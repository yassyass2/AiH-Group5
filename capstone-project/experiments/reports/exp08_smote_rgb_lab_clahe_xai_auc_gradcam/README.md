# Experiment 08 Report

Experiment 08 evaluates EfficientNetB0 on APTOS 2019 with RGB plus LAB-CLAHE preprocessing and conservative SMOTE.

The important design choice is that the dataset is split before oversampling. SMOTE is applied only to the training split, while validation and test data remain untouched. Minority disease classes are raised toward the Moderate class count instead of the No-DR majority count.

Held-out test results:

- Accuracy: 0.8091
- Macro F1: 0.6329
- Weighted F1: 0.8018
- Quadratic weighted kappa: 0.8866
- OVR macro AUC: 0.9231

APTOS 2019 does not include lesion masks. Grad-CAM examples are included for qualitative review, but quantitative localization metrics such as Pointing Game and IoU require a pixel-annotated lesion dataset.
