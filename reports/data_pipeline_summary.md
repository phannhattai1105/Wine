# Data Pipeline Summary -- Wine PCA Project

> Cập nhật: 2026-08-18 14:31

## 1. Thông tin Dataset

| Thông số | Giá trị |
|----------|---------|
| File gốc | data/raw/Wine.csv |
| Tổng dòng | 181 |
| Số feature | 13 (tất cả numerical) |
| Cột nhãn | Customer_Segment (3 lớp: 1, 2, 3) |

## 2. Split

| Tập | Số dòng | Tỉ lệ |
|-----|--------:|------:|
| Train | 127 | ~70% |
| Val   | 27  | ~15% |
| Test  | 27  | ~15% |

Stratified Split (giữ tỷ lệ nhãn ở cả 3 tập), random_state=42.

## 3. EDA -- Vấn đề dữ liệu (raw)

| Vấn đề | Số lượng | Chi tiết |
|--------|:--------:|---------|
| Missing values | 5 ô | Ash (3), Hue (2) |
| Duplicate rows | 1 dòng | Trong tập Train |
| Outlier (IQR)  | 28 giá trị | Magnesium, Proline rõ nhất |

## 4. Feature Engineering

| Bước | Chiến lược |
|------|------------|
| Drop duplicate | `drop_duplicates()` |
| Impute missing | Median từ Train (anti-leakage) |
| Clip outlier | IQR bounds từ Train (anti-leakage) |

## 5. Encoding

Không có cột categorical. `encode_if_needed()` trả về DataFrame nguyên vẹn.

## 6. Standardization

Phương pháp: z-score (population std, ddof=0).
Mean và std chỉ fit từ Train. Val/Test chỉ được transform.

| Tập | mean (sau scale) | std (sau scale) |
|-----|----------------:|----------------:|
| X_train_scaled | 0.000000 | 1.000000 |
| X_val_scaled   | 0.043013 | 0.986547 |
| X_test_scaled  | -0.021589 | 1.025015 |

## 7. Các cặp feature tương quan mạnh (|r| > 0.7)

| Feature A | Feature B | r |
|-----------|-----------|---:|
| Total_Phenols | Flavanoids | 0.860 |
| Flavanoids | OD280 | 0.798 |

> Tương quan mạnh giữa nhiều feature -> PCA có thể nén chiều hiệu quả.

## 8. Output files

| File | Mô tả |
|------|-------|
| data/processed/train_raw.csv | Train raw (sau split) |
| data/processed/val_raw.csv   | Val raw (sau split) |
| data/processed/test_raw.csv  | Test raw (sau split) |
| data/processed/preprocessing_params.pkl | clean_params + scaler |
| reports/figures/missing_values_bar.png  | Bar chart missing |
| reports/figures/histogram_features.png  | Histogram 13 features |
| reports/figures/correlation_heatmap.png | Correlation matrix |
| reports/figures/boxplot_features_before.png | Boxplot trước xử lý |
| reports/figures/boxplot_features_after.png  | Boxplot sau xử lý |
| reports/figures/boxplot_before_after.png    | So sánh trước/sau |
