# PCA Wine Project

Đồ án môn học **Thực hành thuật toán PCA** — Wine Dataset.

## Cấu trúc thư mục

```
project-pca-wine/
├── data/
│   ├── raw/Wine.csv                          ← dataset gốc (181 dòng, có lỗi demo)
│   └── processed/
│       ├── train_raw.csv                     ← sau split, chưa clean
│       ├── val_raw.csv
│       ├── test_raw.csv
│       └── preprocessing_params.pkl          ← params fit từ train
├── notebooks/
│   ├── 01_split.ipynb                        ← Data Pipeline (split)
│   ├── 02_eda_preprocessing.ipynb            ← Data Pipeline (EDA + preprocessing)
│   ├── 03_pca_scratch_evaluation.ipynb       ← PCA Model (Đăng)
│   └── 04_compare_with_sklearn.ipynb         ← So sánh với sklearn PCA (Đăng)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                      ← Public API của Data Pipeline
│   └── pca_scratch/                          ← Cài đặt PCA từ scratch (Đăng)
├── reports/
│   ├── data_pipeline_summary.md
│   └── figures/
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy

```bash
# Bước 1 — Split (chạy trước tiên)
jupyter notebook notebooks/01_split.ipynb

# Bước 2 — EDA & Preprocessing
jupyter notebook notebooks/02_eda_preprocessing.ipynb
```

## Public API (`src/preprocessing.py`)

| Hàm | Mô tả |
|-----|-------|
| `load_raw_data(path)` | Đọc Wine.csv |
| `split_data(df, ...)` | Stratified split → (train, val, test) |
| `fit_clean_params(train_df)` | Học median/IQR từ train |
| `clean(df, params)` | Drop dup / impute / clip outlier |
| `fit_scaler(train_clean_df)` | Fit StandardScaler từ train |
| `scale(df, scaler)` | Chuẩn hoá → np.ndarray |

## Nguyên tắc chống Leakage

Mọi tham số (median, IQR, mean/std) chỉ được **fit từ Train**.  
Val/Test chỉ được **transform** — không fit lại.
