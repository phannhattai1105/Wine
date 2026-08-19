# Data Pipeline Summary -- Wine PCA Project

> Cap nhat: 2026-08-19 10:58

## 1. Thong tin Dataset

| Thong so | Gia tri |
|----------|---------|
| File goc | data/raw/Wine.csv |
| Tong dong | 181 |
| So feature | 13 (tat ca numerical) |
| Cot nhan | Customer_Segment (3 lop: 1, 2, 3) |

## 2. Split

| Tap | So dong | Ti le |
|-----|--------:|------:|
| Train | 127 | ~70% |
| Val   | 27  | ~15% |
| Test  | 27  | ~15% |

Stratified Split (giu ty le nhan o ca 3 tap), random_state=42.

## 3. EDA -- Van de du lieu (raw)

| Van de | So luong | Chi tiet |
|--------|:--------:|---------|
| Missing values | 5 o | Ash (3), Hue (2) |
| Duplicate rows | 1 dong | Trong tap Train |
| Outlier (IQR)  | 28 gia tri | Magnesium, Proline ro nhat |

## 4. Feature Engineering

| Buoc | Chien luoc |
|------|------------|
| Drop duplicate | `drop_duplicates()` |
| Impute missing | Median tu Train (anti-leakage) |
| Clip outlier | IQR bounds tu Train (anti-leakage) |

## 5. Encoding

Khong co cot categorical. `encode_if_needed()` tra ve DataFrame nguyen ven.

## 6. Standardization

Phuong phap: z-score (population std, ddof=0).
Mean va std chi fit tu Train. Val/Test chi duoc transform.

| Tap | mean (sau scale) | std (sau scale) |
|-----|----------------:|----------------:|
| X_train_scaled | 0.000000 | 1.000000 |
| X_val_scaled   | 0.043013 | 0.986547 |
| X_test_scaled  | -0.021589 | 1.025015 |

## 7. Cac cap feature tuong quan manh (|r| > 0.7)

| Feature A | Feature B | r |
|-----------|-----------|---:|
| Total_Phenols | Flavanoids | 0.860 |
| Flavanoids | OD280 | 0.798 |

> Tuong quan manh giua nhieu feature -> PCA co the nen chieu hieu qua.

## 8. Output files

| File | Mo ta |
|------|-------|
| data/processed/train_raw.csv | Train raw (sau split) |
| data/processed/val_raw.csv   | Val raw (sau split) |
| data/processed/test_raw.csv  | Test raw (sau split) |
| data/processed/preprocessing_params.pkl | clean_params + scaler |
| reports/figures/missing_values_bar.png  | Bar chart missing |
| reports/figures/histogram_features.png  | Histogram 13 features |
| reports/figures/correlation_heatmap.png | Correlation matrix |
| reports/figures/boxplot_features_before.png | Boxplot truoc xu ly |
| reports/figures/boxplot_features_after.png  | Boxplot sau xu ly |
| reports/figures/boxplot_before_after.png    | So sanh truoc/sau |
| reports/figures/class_distribution_split.png | Phan bo nhan Train/Val/Test |
| reports/figures/violin_by_class.png          | Violin plot theo lop |
| reports/figures/pairplot_top_features.png    | Pairplot 5 feature noi bat |
| reports/figures/radar_class_profile.png      | Radar ho so trung binh theo lop |
| reports/figures/outlier_count_before_after.png | So sanh outlier truoc/sau |
| reports/figures/class_separation_preview.png | Scatter 2 feature tuong quan manh nhat |
