import pickle
import pathlib
import numpy as np
import pandas as pd

# Danh sách mặc định 13 cột feature của Wine dataset
WINE_NUMERICAL_COLS: list[str] = [
    "Alcohol", "Malic_Acid", "Ash", "Ash_Alcanity", "Magnesium",
    "Total_Phenols", "Flavanoids", "Nonflavanoid_Phenols",
    "Proanthocyanins", "Color_Intensity", "Hue", "OD280", "Proline",
]
WINE_TARGET_COL: str = "Customer_Segment"


# 1. LOAD
def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# 2. SPLIT  (phải gọi TRƯỚC mọi bước fit tham số để tránh data leakage)
def split_data(
    df: pd.DataFrame,
    target_col: str = "Customer_Segment",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if test_size + val_size >= 1.0:
        raise ValueError(
            f"test_size ({test_size}) + val_size ({val_size}) phải < 1.0"
        )

    rng = np.random.default_rng(random_state)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for class_label, group in df.groupby(target_col, sort=True):
        idx = group.index.to_numpy()
        idx = rng.permutation(idx)               # xáo trộn trong nhóm
        n = len(idx)
        n_test = round(n * test_size)
        n_val  = round(n * val_size)
        # n_train = n - n_test - n_val (phần còn lại)

        test_idx.extend(idx[:n_test].tolist())
        val_idx.extend(idx[n_test : n_test + n_val].tolist())
        train_idx.extend(idx[n_test + n_val :].tolist())

    # Xáo trộn lại mỗi tập để không bị sắp theo nhóm nhãn liên tiếp
    train_idx = rng.permutation(train_idx).tolist()
    val_idx   = rng.permutation(val_idx).tolist()
    test_idx  = rng.permutation(test_idx).tolist()

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df   = df.loc[val_idx].reset_index(drop=True)
    test_df  = df.loc[test_idx].reset_index(drop=True)

    # In nhanh để kiểm tra
    total = len(df)
    print("=" * 55)
    print(f"[split_data] Tổng: {total} dòng | "
          f"random_state={random_state}")
    print(f"  Train : {len(train_df):4d} dòng  "
          f"({len(train_df)/total*100:.1f}%)")
    print(f"  Val   : {len(val_df):4d} dòng  "
          f"({len(val_df)/total*100:.1f}%)")
    print(f"  Test  : {len(test_df):4d} dòng  "
          f"({len(test_df)/total*100:.1f}%)")
    for name, subset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = (
            subset[target_col]
            .value_counts(normalize=True)
            .sort_index()
            .apply(lambda x: f"{x:.1%}")
            .to_dict()
        )
        print(f"  {name} phân bố {target_col}: {dist}")
    print("=" * 55)

    return train_df, val_df, test_df


# 3. FIT CLEANING PARAMS  (chỉ fit từ train)
def fit_clean_params(
    train_df: pd.DataFrame,
    numerical_cols: list[str],
) -> dict:

    # Drop duplicate trước khi tính tham số để tránh bias
    train_dedup = train_df.drop_duplicates().copy()

    median_map: dict[str, float] = {}
    iqr_bounds_map: dict[str, tuple[float, float]] = {}

    for col in numerical_cols:
        series = train_dedup[col].dropna()   # bỏ NaN khi tính tham số

        # Median (dùng để impute missing)
        median_map[col] = float(series.median())

        # IQR bounds (dùng để clip outlier)
        q1  = float(series.quantile(0.25))
        q3  = float(series.quantile(0.75))
        iqr = q3 - q1
        iqr_bounds_map[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    params = {
        "median": median_map,
        "iqr_bounds": iqr_bounds_map,
    }
    print(f"[fit_clean_params] Đã học tham số cho {len(numerical_cols)} cột "
          f"(từ {len(train_dedup)} dòng sau khi drop duplicate).")
    return params


# 4. CLEAN  (apply lên train / val / test — pure transform, không fit lại)
def clean(
    df: pd.DataFrame,
    params: dict,
    numerical_cols: list[str],
) -> pd.DataFrame:

    result = df.copy()

    # ── Bước 1: Drop duplicate ────────────────────────────────────────────
    n_before = len(result)
    result = result.drop_duplicates()
    n_dup_removed = n_before - len(result)
    print(f"  [clean] Bước 1 – Duplicate đã xoá    : {n_dup_removed} dòng "
          f"({n_before} → {len(result)})")

    # ── Bước 2: Impute missing bằng median (từ train) ─────────────────────
    n_missing_before = int(result[numerical_cols].isnull().sum().sum())
    for col in numerical_cols:
        if col in params["median"]:
            result[col] = result[col].fillna(params["median"][col])
    n_missing_after = int(result[numerical_cols].isnull().sum().sum())
    print(f"  [clean] Bước 2 – Missing đã điền     : "
          f"{n_missing_before - n_missing_after} ô "
          f"(còn lại {n_missing_after})")

    # ── Bước 3: Clip outlier theo IQR bounds (từ train) ───────────────────
    n_outlier_total = 0
    for col in numerical_cols:
        if col in params["iqr_bounds"]:
            low, high = params["iqr_bounds"][col]
            n_out = int(((result[col] < low) | (result[col] > high)).sum())
            n_outlier_total += n_out
            result[col] = result[col].clip(lower=low, upper=high)
    print(f"  [clean] Bước 3 – Outlier đã clip     : {n_outlier_total} giá trị")
    print(f"  [clean] Shape sau clean               : {result.reset_index(drop=True).shape}")

    return result.reset_index(drop=True)


# 5. ENCODE IF NEEDED  (dự phòng — Wine dataset không cần dùng)
def encode_if_needed(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:

    # Phát hiện tự động nếu không truyền danh sách
    if not categorical_cols:
        categorical_cols = [
            c for c in df.columns
            if pd.api.types.is_object_dtype(df[c])
            or isinstance(df[c].dtype, pd.CategoricalDtype)
        ]

    if not categorical_cols:
        print("[encode_if_needed] Không có cột categorical, bỏ qua encoding.")
        return df

    print(f"[encode_if_needed] Encoding {len(categorical_cols)} cột: "
          f"{categorical_cols}")
    # TODO (anti-leakage): lưu danh sách cột sau get_dummies trên Train,
    # rồi df_val = df_val.reindex(columns=train_dummies_cols, fill_value=0)
    result = pd.get_dummies(df, columns=categorical_cols, dtype=np.uint8)
    print(f"[encode_if_needed] Shape sau encoding: {result.shape}")
    return result

# 6. FIT SCALER  (chỉ fit từ train_clean — KHÔNG dùng sklearn)
def fit_scaler(
    train_clean_df: pd.DataFrame,
    numerical_cols: list[str],
) -> dict:
    
    feature_df = train_clean_df[numerical_cols]
    mean = feature_df.mean()                    # pandas mặc định ddof=1 cho std,
    std  = feature_df.std(ddof=0)               # nhưng ta dùng ddof=0 (population std)

    print(f"[fit_scaler] Đã fit scaler từ {len(train_clean_df)} mẫu train.")
    print(f"  mean (5 cột đầu): {mean.head().to_dict()}")
    print(f"  std  (5 cột đầu): {std.head().to_dict()}")

    return {"mean": mean, "std": std}


# 7. SCALE  (apply lên train / val / test — pure transform, không fit lại)
def scale(
    df: pd.DataFrame,
    scaler: dict,
    numerical_cols: list[str],
) -> np.ndarray:

    X = df[numerical_cols].copy()

    mean = scaler["mean"]   # pd.Series học từ Train
    std  = scaler["std"]    # pd.Series học từ Train (ddof=0)

    # Xử lý std == 0 để tránh NaN (cột hằng số → gán 0 sau khi trừ mean)
    std_safe = std.replace(0, np.nan)                  # tạm thay 0 bằng NaN
    X_scaled = (X - mean) / std_safe                   # NaN/NaN → NaN ở cột hằng
    X_scaled = X_scaled.fillna(0.0)                    # thay NaN về 0 cho cột hằng

    return X_scaled.to_numpy()



def save_pipeline_params(
    clean_params: dict,
    scaler: dict,
    path: str,
) -> None:

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clean_params": clean_params,
        "scaler": scaler,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[save_pipeline_params] Đã lưu tham số → {path}")


# 9. LOAD PIPELINE PARAMS
def load_pipeline_params(path: str) -> dict:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    print(f"[load_pipeline_params] Đã tải tham số ← {path}")
    return payload


# SELF-TEST  — chạy thử toàn bộ pipeline trên Wine.csv
if __name__ == "__main__":
    import sys
    # Buộc stdout dùng UTF-8 để tránh UnicodeEncodeError trên Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Đường dẫn tương đối từ vị trí file này (src/) lên project root
    _SRC_DIR  = pathlib.Path(__file__).parent
    _ROOT_DIR = _SRC_DIR.parent
    _DATA_RAW = _ROOT_DIR / "data" / "raw" / "Wine.csv"
    _PARAMS_PKL = _ROOT_DIR / "data" / "processed" / "preprocessing_params.pkl"

    print("\n" + "=" * 60)
    print("  SELF-TEST: Data Pipeline -- Wine PCA Project")
    print("=" * 60)

    # -- 1. Load --
    print("\n>> Buoc 1 -- Load raw data")
    df_raw = load_raw_data(str(_DATA_RAW))
    print(f"  Shape raw: {df_raw.shape}")
    print(f"  Cột: {list(df_raw.columns)}")
    print(f"  Missing:\n{df_raw.isnull().sum()[df_raw.isnull().sum() > 0]}")
    print(f"  Duplicate: {df_raw.duplicated().sum()} dòng")

    # -- 2. Split --
    print("\n>> Buoc 2 -- Stratified Split (70 / 15 / 15)")
    train_df, val_df, test_df = split_data(
        df_raw,
        target_col=WINE_TARGET_COL,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
    )

    # -- 3. Fit cleaning params tu train --
    print("\n>> Buoc 3 -- Fit clean params (tu train)")
    clean_params = fit_clean_params(train_df, WINE_NUMERICAL_COLS)
    print(f"  Ví dụ median Ash  : {clean_params['median'].get('Ash'):.4f}")
    print(f"  Ví dụ IQR Proline : {clean_params['iqr_bounds'].get('Proline')}")

    # -- 4. Clean train / val / test --
    print("\n>> Buoc 4 -- Clean")
    print("  [Train]")
    train_clean = clean(train_df, clean_params, WINE_NUMERICAL_COLS)
    print("  [Val]")
    val_clean   = clean(val_df,   clean_params, WINE_NUMERICAL_COLS)
    print("  [Test]")
    test_clean  = clean(test_df,  clean_params, WINE_NUMERICAL_COLS)

    print(f"\n  Shape sau clean — Train:{train_clean.shape}, "
          f"Val:{val_clean.shape}, Test:{test_clean.shape}")

    # -- 5. Encode (du phong) --
    print("\n>> Buoc 5 -- Encode if needed")
    train_enc = encode_if_needed(train_clean)
    val_enc   = encode_if_needed(val_clean)
    test_enc  = encode_if_needed(test_clean)

    # -- 6. Fit scaler tu train --
    print("\n>> Buoc 6 -- Fit scaler (tu train_clean)")
    scaler = fit_scaler(train_enc, WINE_NUMERICAL_COLS)

    # -- 7. Scale train / val / test --
    print("\n>> Buoc 7 -- Scale")
    X_train = scale(train_enc, scaler, WINE_NUMERICAL_COLS)
    X_val   = scale(val_enc,   scaler, WINE_NUMERICAL_COLS)
    X_test  = scale(test_enc,  scaler, WINE_NUMERICAL_COLS)

    print(f"  X_train : {X_train.shape} | mean≈{X_train.mean():.4f} | std≈{X_train.std():.4f}")
    print(f"  X_val   : {X_val.shape}   | mean≈{X_val.mean():.4f} | std≈{X_val.std():.4f}")
    print(f"  X_test  : {X_test.shape}  | mean≈{X_test.mean():.4f} | std≈{X_test.std():.4f}")

    # -- 8. Save params --
    print("\n>> Buoc 8 -- Save pipeline params")
    save_pipeline_params(clean_params, scaler, str(_PARAMS_PKL))

    # -- 9. Load lai va kiem tra --
    print("\n>> Buoc 9 -- Load pipeline params & kiem tra")
    loaded = load_pipeline_params(str(_PARAMS_PKL))
    assert set(loaded.keys()) == {"clean_params", "scaler"}, "Load sai cấu trúc!"
    print("  ✅ Load OK — keys:", list(loaded.keys()))

    print("\n" + "=" * 60)
    print("  [OK] SELF-TEST PASSED -- Toan bo pipeline hoat dong dung.")
    print("=" * 60 + "\n")
