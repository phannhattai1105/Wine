"""
src/preprocessing.py
====================
Data Pipeline module cho đồ án PCA – Wine Dataset.

Module này được import bởi:
  - notebooks/01_split.ipynb              (load_raw_data, split_data)
  - notebooks/02_eda_preprocessing.ipynb  (fit_clean_params, clean,
                                           encode_if_needed, fit_scaler, scale)
  - src/pca_scratch/ (của Đăng) thông qua tất cả các hàm công khai bên dưới.

RÀNG BUỘC QUAN TRỌNG — KHÔNG dùng scikit-learn
------------------------------------------------
Toàn bộ logic trong module này được cài đặt thuần bằng numpy/pandas:
  - Stratified Split    : tự cài đặt bằng np.random.default_rng + groupby
  - Standardization     : tự tính mean/std từ train, lưu vào dict
  - Impute / IQR / Clip : tự tính bằng pandas/numpy

NGUYÊN TẮC CHỐNG LEAKAGE
--------------------------
Mọi tham số thống kê (median để impute, ngưỡng IQR để detect outlier,
mean/std để chuẩn hoá) đều CHỈ được học (fit) từ tập Train.
Val và Test chỉ được transform theo tham số đã học — KHÔNG fit lại.

Thứ tự thực thi thực tế:
  1. load_raw_data()
  2. split_data()          <-- Split TRƯỚC
  3. fit_clean_params()    <-- Fit từ train_df
  4. clean()               <-- Apply lên train / val / test
  5. encode_if_needed()    <-- Dự phòng, Wine không cần
  6. fit_scaler()          <-- Fit từ train_clean_df
  7. scale()               <-- Apply lên train / val / test

Hàm công khai (public API – không đổi tên, Đăng sẽ import trực tiếp):
  - load_raw_data(path) -> pd.DataFrame
  - split_data(df, target_col, test_size, val_size, random_state)
        -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
  - fit_clean_params(train_df, numerical_cols) -> dict
  - clean(df, params, numerical_cols) -> pd.DataFrame
  - encode_if_needed(df, categorical_cols) -> pd.DataFrame
  - fit_scaler(train_clean_df, numerical_cols) -> dict
  - scale(df, scaler, numerical_cols) -> np.ndarray
  - save_pipeline_params(clean_params, scaler, path) -> None
  - load_pipeline_params(path) -> dict
"""

import pickle
import pathlib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Danh sách mặc định 13 cột feature của Wine dataset
# ---------------------------------------------------------------------------
WINE_NUMERICAL_COLS: list[str] = [
    "Alcohol", "Malic_Acid", "Ash", "Ash_Alcanity", "Magnesium",
    "Total_Phenols", "Flavanoids", "Nonflavanoid_Phenols",
    "Proanthocyanins", "Color_Intensity", "Hue", "OD280", "Proline",
]
WINE_TARGET_COL: str = "Customer_Segment"


# ===========================================================================
# 1. LOAD
# ===========================================================================

def load_raw_data(path: str) -> pd.DataFrame:
    """Đọc file CSV raw và trả về DataFrame gốc chưa xử lý.

    Args:
        path: Đường dẫn tới file CSV (tuyệt đối hoặc tương đối từ nơi gọi).
              Ví dụ: ``"data/raw/Wine.csv"`` hoặc ``"../data/raw/Wine.csv"``.

    Returns:
        DataFrame với toàn bộ cột gốc, index 0-based (reset tự động từ CSV).
        Không có bất kỳ bước làm sạch hay biến đổi nào được thực hiện.

    Raises:
        FileNotFoundError: Nếu ``path`` không tồn tại.
    """
    df = pd.read_csv(path)
    return df


# ===========================================================================
# 2. SPLIT  (phải gọi TRƯỚC mọi bước fit tham số để tránh data leakage)
# ===========================================================================

def split_data(
    df: pd.DataFrame,
    target_col: str = "Customer_Segment",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tự cài đặt Stratified Split bằng numpy thuần — KHÔNG dùng sklearn.

    **Quan trọng – Anti-Leakage:**
    Hàm này PHẢI được gọi TRƯỚC mọi bước fit tham số (fit_clean_params,
    fit_scaler). Lý do: nếu làm sạch/chuẩn hoá toàn bộ dataset trước khi
    split, median/IQR/mean/std sẽ bị "nhiễm" thông tin từ Val và Test,
    dẫn đến data leakage — mô hình trông có vẻ tốt hơn thực tế.

    **Thuật toán Stratified Split tự cài đặt:**
        1. Dùng ``np.random.default_rng(random_state)`` tạo generator ngẫu
           nhiên tái lập được (thay thế ``random_state`` của sklearn).
        2. Nhóm các dòng theo giá trị ``target_col`` (``df.groupby``).
        3. Với mỗi nhóm, xáo trộn index bằng ``rng.permutation()``, rồi
           cắt theo tỉ lệ::

               n_test  = round(n * test_size)
               n_val   = round(n * val_size)
               n_train = n - n_test - n_val

        4. Gộp index của từng tập qua toàn bộ nhóm nhãn.
        5. Xáo trộn lại mỗi tập một lần nữa (``rng.permutation``) để tránh
           dữ liệu bị sắp xếp liên tiếp theo nhóm nhãn, rồi ``reset_index``.

    Kết quả đảm bảo tỉ lệ nhãn (class distribution) được bảo toàn ở cả
    ba tập Train / Val / Test.

    Args:
        df: DataFrame đầu vào (raw, chưa xử lý).
        target_col: Tên cột nhãn dùng để stratify. Mặc định ``"Customer_Segment"``.
        test_size: Tỉ lệ dành cho tập Test trên tổng dataset. Mặc định 0.15.
        val_size: Tỉ lệ dành cho tập Val trên tổng dataset. Mặc định 0.15.
        random_state: Seed để tái lập kết quả. Mặc định 42.

    Returns:
        Tuple ``(train_df, val_df, test_df)`` — mỗi DataFrame giữ nguyên
        tất cả cột gốc (bao gồm ``target_col``), index reset về 0-based.

    Raises:
        ValueError: Nếu ``test_size + val_size >= 1.0``.
    """
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


# ===========================================================================
# 3. FIT CLEANING PARAMS  (chỉ fit từ train)
# ===========================================================================

def fit_clean_params(
    train_df: pd.DataFrame,
    numerical_cols: list[str],
) -> dict:
    """Học các tham số làm sạch CHỈ từ tập Train — KHÔNG fit trên Val/Test.

    Tham số được học:
        - ``median``: median của mỗi cột numerical (để impute missing).
        - ``iqr_bounds``: ngưỡng dưới/trên IQR của mỗi cột (để clip outlier).
          Công thức: lower = Q1 − 1.5·IQR, upper = Q3 + 1.5·IQR.

    Lưu ý: Duplicate trong ``train_df`` được drop TRƯỚC khi tính tham số
    để tránh bias (các dòng lặp sẽ kéo lệch median nếu giữ lại).

    Args:
        train_df: Tập Train raw (chưa impute, chưa clip, có thể có duplicate).
        numerical_cols: Danh sách tên cột số để tính tham số.
                        Cột ``Customer_Segment`` nên được loại khỏi danh sách này.

    Returns:
        Dict với cấu trúc::

            {
                "median": {
                    "Ash": 2.36,
                    "Hue": 0.96,
                    ...
                },
                "iqr_bounds": {
                    "Magnesium": (70.25, 138.75),
                    "Proline"  : (415.0, 1185.0),
                    ...
                }
            }
    """
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


# ===========================================================================
# 4. CLEAN  (apply lên train / val / test — pure transform, không fit lại)
# ===========================================================================

def clean(
    df: pd.DataFrame,
    params: dict,
    numerical_cols: list[str],
) -> pd.DataFrame:
    """Làm sạch DataFrame bằng tham số đã học từ Train.

    Hàm này là **pure transform** — chỉ áp dụng tham số có sẵn trong
    ``params``, tuyệt đối KHÔNG tính lại median/IQR từ ``df`` truyền vào
    (dù ``df`` là Val hay Test).

    Thứ tự xử lý:
        1. **Drop duplicate** — xoá trước để không mất dòng missing cần
           minh hoạ do bị dedup nhầm thứ tự.
        2. **Fillna** — impute missing bằng ``params["median"]`` (học từ train).
        3. **Clip outlier** — clip về ``params["iqr_bounds"]`` (học từ train),
           KHÔNG xoá dòng để giữ nguyên số lượng mẫu.

    Args:
        df: DataFrame cần làm sạch (train, val, hoặc test).
        params: Dict tham số trả về từ :func:`fit_clean_params`.
        numerical_cols: Danh sách cột số cần xử lý.

    Returns:
        DataFrame mới đã làm sạch, ``reset_index(drop=True)``.
        Cột ``Customer_Segment`` và các cột khác ngoài ``numerical_cols``
        được giữ nguyên, không bị chỉnh sửa.
    """
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


# ===========================================================================
# 5. ENCODE IF NEEDED  (dự phòng — Wine dataset không cần dùng)
# ===========================================================================

def encode_if_needed(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:
    """One-Hot Encode các cột categorical nếu có — dự phòng cho dataset mở rộng.

    Dataset Wine hiện tại toàn bộ feature đã là kiểu số (numerical), nên
    hàm này sẽ trả về ``df`` nguyên vẹn và in thông báo bỏ qua. Tuy nhiên
    logic được viết sẵn để xử lý đúng nếu sau này dataset có thêm cột
    categorical.

    **Lưu ý Anti-Leakage:** Hàm hiện dùng ``pd.get_dummies()`` trực tiếp
    trên ``df``, nghĩa là các dummy column được tạo từ chính tập đó.
    Nếu Val/Test có giá trị categorical chưa từng xuất hiện ở Train, cột
    sẽ bị thiếu. Để xử lý đúng hơn, cần lưu lại danh sách cột sau
    ``get_dummies`` trên Train và ``reindex`` Val/Test theo đó — xem
    comment TODO bên dưới.

    Args:
        df: DataFrame đầu vào.
        categorical_cols: Danh sách tên cột cần encode. Nếu ``None`` hoặc
            rỗng, hàm tự phát hiện cột ``dtype == object`` hoặc ``category``.

    Returns:
        DataFrame sau khi One-Hot Encode (hoặc nguyên vẹn nếu không có
        cột categorical). Boolean dummy được giữ nguyên kiểu bool/uint8
        của ``pd.get_dummies``.
    """
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


# ===========================================================================
# 6. FIT SCALER  (chỉ fit từ train_clean — KHÔNG dùng sklearn)
# ===========================================================================

def fit_scaler(
    train_clean_df: pd.DataFrame,
    numerical_cols: list[str],
) -> dict:
    """Tính mean và std của từng cột feature CHỈ từ tập Train đã clean.

    Tự cài đặt Standardization bằng pandas/numpy — KHÔNG dùng
    ``sklearn.preprocessing.StandardScaler``.

    Công thức giống với StandardScaler mặc định của sklearn (``ddof=0``,
    tức population std) để kết quả khớp khi Đăng so sánh ở notebook 04.

    Args:
        train_clean_df: Tập Train đã qua bước :func:`clean`.
        numerical_cols: Danh sách cột feature số (không bao gồm cột nhãn).

    Returns:
        Dict custom scaler::

            {
                "mean": pd.Series({"Alcohol": 13.0, "Ash": 2.36, ...}),
                "std" : pd.Series({"Alcohol":  0.81, "Ash": 0.27, ...})
            }

        Chỉ học từ ``train_clean_df`` — truyền dict này vào :func:`scale`
        để transform Val/Test mà không tính lại tham số.
    """
    feature_df = train_clean_df[numerical_cols]
    mean = feature_df.mean()                    # pandas mặc định ddof=1 cho std,
    std  = feature_df.std(ddof=0)               # nhưng ta dùng ddof=0 (population std)

    print(f"[fit_scaler] Đã fit scaler từ {len(train_clean_df)} mẫu train.")
    print(f"  mean (5 cột đầu): {mean.head().to_dict()}")
    print(f"  std  (5 cột đầu): {std.head().to_dict()}")

    return {"mean": mean, "std": std}


# ===========================================================================
# 7. SCALE  (apply lên train / val / test — pure transform, không fit lại)
# ===========================================================================

def scale(
    df: pd.DataFrame,
    scaler: dict,
    numerical_cols: list[str],
) -> np.ndarray:
    """Chuẩn hoá các cột feature theo công thức z-score dùng scaler đã fit.

    Công thức: ``X_scaled = (X − mean_train) / std_train``

    Hàm này là **pure transform** — LUÔN dùng ``scaler["mean"]`` và
    ``scaler["std"]`` đã học từ Train, tuyệt đối KHÔNG tính lại mean/std
    từ ``df`` truyền vào (dù đó là Val hay Test). Nguyên tắc này giống
    hệt ``transform()`` của sklearn: chỉ áp dụng tham số đã fit, không
    re-fit.

    Args:
        df: DataFrame đã qua :func:`clean` (train, val, hoặc test).
        scaler: Dict ``{"mean": pd.Series, "std": pd.Series}`` trả về
                từ :func:`fit_scaler` — học từ Train.
        numerical_cols: Danh sách cột feature cần chuẩn hoá.

    Returns:
        numpy array shape ``(n_samples, n_features)`` đã chuẩn hoá.
        Cột ``Customer_Segment`` KHÔNG có mặt trong output này.

    Notes:
        Nếu một cột có ``std == 0`` (hằng số), kết quả cột đó sẽ là
        ``0.0`` thay vì ``NaN`` (tránh lỗi chia-cho-0).
    """
    # Lấy đúng các cột feature, KHÔNG dùng lại mean/std của df này
    # mà dùng scaler["mean"] / scaler["std"] từ Train — chống leakage
    X = df[numerical_cols].copy()

    mean = scaler["mean"]   # pd.Series học từ Train
    std  = scaler["std"]    # pd.Series học từ Train (ddof=0)

    # Xử lý std == 0 để tránh NaN (cột hằng số → gán 0 sau khi trừ mean)
    std_safe = std.replace(0, np.nan)                  # tạm thay 0 bằng NaN
    X_scaled = (X - mean) / std_safe                   # NaN/NaN → NaN ở cột hằng
    X_scaled = X_scaled.fillna(0.0)                    # thay NaN về 0 cho cột hằng

    return X_scaled.to_numpy()


# ===========================================================================
# 8. SAVE PIPELINE PARAMS
# ===========================================================================

def save_pipeline_params(
    clean_params: dict,
    scaler: dict,
    path: str,
) -> None:
    """Lưu cả clean_params và scaler vào một file .pkl bằng pickle.

    Args:
        clean_params: Dict trả về từ :func:`fit_clean_params`.
        scaler: Dict trả về từ :func:`fit_scaler`.
        path: Đường dẫn file .pkl để lưu.
              Ví dụ: ``"data/processed/preprocessing_params.pkl"``.

    Returns:
        None. File .pkl được ghi ra đĩa.
    """
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clean_params": clean_params,
        "scaler": scaler,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[save_pipeline_params] Đã lưu tham số → {path}")


# ===========================================================================
# 9. LOAD PIPELINE PARAMS
# ===========================================================================

def load_pipeline_params(path: str) -> dict:
    """Đọc lại file .pkl đã lưu bởi :func:`save_pipeline_params`.

    Args:
        path: Đường dẫn file .pkl.

    Returns:
        Dict với hai key::

            {
                "clean_params": { "median": {...}, "iqr_bounds": {...} },
                "scaler":       { "mean": pd.Series, "std": pd.Series  }
            }

    Raises:
        FileNotFoundError: Nếu ``path`` không tồn tại.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    print(f"[load_pipeline_params] Đã tải tham số ← {path}")
    return payload


# ===========================================================================
# SELF-TEST  — chạy thử toàn bộ pipeline trên Wine.csv
# ===========================================================================

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
