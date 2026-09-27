# Dataset

This project uses the **E Commerce Dataset** from Kaggle:
https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

The raw file (`E_Commerce_Dataset.xlsx`) is not committed to this repository (Kaggle's terms
don't permit redistribution, and it's good practice to keep raw data out of version control).

## To reproduce

1. Download `E Commerce Dataset.xlsx` from the Kaggle link above.
2. Rename it to `E_Commerce_Dataset.xlsx` and place it in this folder (`data/raw/`).
3. It contains two sheets: `Data Dict` (column descriptions) and `E Comm` (the actual data,
   5,630 rows × 20 columns) — the pipeline reads the `E Comm` sheet.
