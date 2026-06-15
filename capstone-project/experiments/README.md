# Experiments

This folder is for exploratory work that supports the capstone paper but is not part of the production training package.

Use it for Colab notebooks, experiment notes, selected metrics, and figures that are useful for review. Keep raw datasets, model weights, full run folders, and generated preprocessing outputs out of Git. Those belong in local storage, Google Drive, or MLflow artifacts.

## Structure

- `notebooks/`: Colab-ready notebooks. Keep the source readable and clear bulky outputs before committing.
- `reports/`: selected experiment outputs that are small enough to review in Git, such as metric tables, confusion matrices, ROC figures, and short XAI summaries.

## Notebook rules

- Name notebooks with a date and short experiment label: `YYYY-MM-DD-exp-name.ipynb`.
- Put reusable code in `src/dr_grading` when it becomes part of the main pipeline.
- Do not commit Kaggle keys, Drive credentials, model checkpoints, `.npy` files, or full generated artifact directories.
- If a notebook produces paper-relevant figures or tables, export only the selected files under `reports/`.
