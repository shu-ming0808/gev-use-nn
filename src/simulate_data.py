import math
import random
import numpy as np
from scipy.stats import genextreme as gev


# -------------------------------------------------
# 作者 notebook 用的 11 個 quantiles
# 對應百分位:
# 0.01, 0.1, 1, 10, 25, 50, 75, 90, 99, 99.9, 99.99
# -------------------------------------------------
P_SET = np.array(
    [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999],
    dtype=np.float64
)


# -------------------------------------------------
# 設定 random seed
# -------------------------------------------------
def set_seed(seed: int = 111) -> None:
    random.seed(seed)
    np.random.seed(seed)


# -------------------------------------------------
# robust standardization: median / IQR
# -------------------------------------------------
def robust_standardize(x: np.ndarray):
    """
    輸入:
        x: 原始 GEV sample, shape = (n,)

    輸出:
        z: 標準化後 sample
        median: sample median
        iqr: sample IQR
    """
    median = np.median(x)
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25

    if iqr <= 1e-12:
        iqr = 1e-12

    z = (x - median) / iqr
    return z, median, iqr


# -------------------------------------------------
# 單筆資料生成（方便 debug）
# -------------------------------------------------
def generate_one_observation(mu: float, sigma: float, c: float, N: int, rng=None):
    """
    生成單筆 observation，完全對齊作者 notebook 的概念：

    1. sample ~ GEV(c, loc=mu, scale=sigma)
    2. 用 median / IQR 做 robust standardization
    3. 取 11 個 quantiles 當 input X
    4. target Y = [sc_loc, delta, c]

    注意：
        scipy genextreme 的 shape 是 c
        作者 notebook 的 shape_vals 也是 c = -xi
    """
    if rng is None:
        rng = np.random.default_rng()

    sample = gev.rvs(c=c, loc=mu, scale=sigma, size=int(N), random_state=rng)

    z, sample_median, sample_iqr = robust_standardize(sample)

    # 11 維 quantile input
    x_feat = np.percentile(z, P_SET * 100).astype(np.float32)

    # 作者 notebook 的 scaled parameters
    sc_loc = (mu / sample_iqr) - (sample_median / sample_iqr)
    sc_scale = sigma / sample_iqr

    std_sample_min = np.min(z)
    std_sample_max = np.max(z)

    # c > 0 對應 Weibull-type
    # c <= 0 對應 Gumbel / Frechet-type
    is_weibull = c > np.finfo(float).eps

    if is_weibull:
        inside = sc_scale - c * (std_sample_max - sc_loc)
    else:
        inside = sc_scale - c * (std_sample_min - sc_loc)

    if inside <= 1e-12:
        inside = 1e-12

    delta = np.log(inside)

    y_target = np.array([sc_loc, delta, c], dtype=np.float32)

    return (
        sample.astype(np.float32),   # 原始 sample
        z.astype(np.float32),        # standardized sample
        x_feat,                      # input X
        y_target                     # target Y
    )


# -------------------------------------------------
# 作者對齊版資料生成
# -------------------------------------------------
def generate_dataset_author_style(seed: int = 111):
    
    """
    - 總資料數 n = 340000
    - train = 300000
    - valid = 40000
    - N_set = [30, 72, 173, 416, 1000]
    - c ~ Uniform(-1, 0.4)
    - sigma ~ 10^(Uniform(log10(0.1), log10(40)))  # log-uniform
    - mu ~ Uniform(1, 50)

    關鍵：
    每組參數 (mu, sigma, c) 複製到 5 個 sample sizes，
    也就是對每組參數都生成 N=30,72,173,416,1000 這五種資料。

    回傳:
        X          : shape (340000, 11)
        Y          : shape (340000, 3) = [sc_loc, delta, c]
        n_train    : 300000
        n_valid    : 40000
        N_set      : array([  30,   72,  173,  416, 1000])
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # 與作者一致
    n = 340000
    n_train = 300000
    n_valid = 40000

    # 與作者一致
    N_set = np.rint(
        10 ** np.linspace(
            start=math.log(30, 10),
            stop=math.log(1000, 10),
            num=5
        )
    ).astype(int)   # [30, 72, 173, 416, 1000]

    over_all_factor = int(np.rint(n / len(N_set)))   # 68000

    # c = -xi
    shape_vals = rng.uniform(-1.0, 0.4, over_all_factor)

    # sigma: log-uniform
    scale_vals = 10 ** rng.uniform(np.log10(0.1), np.log10(40), over_all_factor)

    # mu: uniform
    loc_vals = rng.uniform(1.0, 50.0, over_all_factor)

    # 與作者一致：每組參數複製 5 次，分配到不同 N
    loc_values = np.hstack([loc_vals] * len(N_set))
    scale_values = np.hstack([scale_vals] * len(N_set))
    shape_values = np.hstack([shape_vals] * len(N_set))
    N_values = np.repeat(N_set, over_all_factor)

    X_list = []
    Y_list = []

    for mu, sigma, c, N in zip(loc_values, scale_values, shape_values, N_values):
        _, _, x_feat, y_target = generate_one_observation(mu, sigma, c, int(N), rng=rng)
        X_list.append(x_feat)
        Y_list.append(y_target)

    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.float32)

    return X, Y, n_train, n_valid, N_set


# -------------------------------------------------
# train / valid split
# -------------------------------------------------
def split_train_valid(X: np.ndarray, Y: np.ndarray, n_train: int = 300000, n_valid: int = 40000):
    """
    作者 notebook 的切法是前 300000 當 train，
    後 40000 當 valid。
    """
    assert X.shape[0] == n_train + n_valid, "X 的筆數不等於 n_train + n_valid"
    assert Y.shape[0] == n_train + n_valid, "Y 的筆數不等於 n_train + n_valid"

    X_train = X[:n_train]
    Y_train = Y[:n_train]

    X_valid = X[n_train:n_train + n_valid]
    Y_valid = Y[n_train:n_train + n_valid]

    return X_train, Y_train, X_valid, Y_valid


# -------------------------------------------------
# 可選：存成 npy
# -------------------------------------------------
def save_dataset_npy(
    X_train, Y_train, X_valid, Y_valid,
    prefix: str = "gev_author_style"
):
    np.save(f"{prefix}_X_train.npy", X_train)
    np.save(f"{prefix}_Y_train.npy", Y_train)
    np.save(f"{prefix}_X_valid.npy", X_valid)
    np.save(f"{prefix}_Y_valid.npy", Y_valid)


# -------------------------------------------------
# 主程式測試
# -------------------------------------------------
if __name__ == "__main__":
    X, Y, n_train, n_valid, N_set = generate_dataset_author_style(seed=111)

    print("========== AUTHOR-STYLE DATASET ==========")
    print("X shape:", X.shape)          # (340000, 11)
    print("Y shape:", Y.shape)          # (340000, 3)
    print("n_train:", n_train)          # 300000
    print("n_valid:", n_valid)          # 40000
    print("N_set:", N_set)              # [  30   72  173  416 1000]

    print("\nFirst 3 X rows:")
    print(X[:3])

    print("\nFirst 3 Y rows:")
    print(Y[:3])

    X_train, Y_train, X_valid, Y_valid = split_train_valid(X, Y, n_train, n_valid)

    print("\n========== SPLIT CHECK ==========")
    print("X_train shape:", X_train.shape)   # (300000, 11)
    print("Y_train shape:", Y_train.shape)   # (300000, 3)
    print("X_valid shape:", X_valid.shape)   # (40000, 11)
    print("Y_valid shape:", Y_valid.shape)   # (40000, 3)
