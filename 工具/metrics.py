# metrics.py
# ===== 函數索引目錄 =====
# align_series             : 對齊兩個報酬序列的共同日期
# max_drawdown             : 最大回撤（依累積淨值計算）
# calc_beta_alpha          : 對基準做線性回歸，回傳 Beta、月α與年化α
# risk_summary             : 產出策略 vs 基準的風險/相對風險總表（AnnRet/Vol/Sharpe/MDD/Calmar/Beta/Corr/TE/IR）
# rolling_beta             : 計算滾動 Beta（預設 24 個月窗）
# format_percent_table     : 友善格式化顯示表格（百分比/小數）
# ========================

import numpy as np
import pandas as pd

# ------------------------------
# 對齊兩個報酬序列
# ------------------------------
def align_series(strategy_ret: pd.Series, bench_ret: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    對齊策略與基準的月報酬（同一索引），並去掉缺值。
    """
    s = strategy_ret.dropna()
    b = bench_ret.dropna()
    idx = s.index.intersection(b.index)
    return s.loc[idx], b.loc[idx]

# ------------------------------
# 最大回撤（依累積淨值）
# ------------------------------
def max_drawdown(r: pd.Series) -> float:
    """
    r: 月度報酬序列
    回傳：最大回撤（負值，例 -0.28）
    """
    if r is None or len(r) == 0:
        return np.nan
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return float(dd.min())

# ------------------------------
# Beta / Alpha（RF=0）
# ------------------------------
def calc_beta_alpha(strategy_ret: pd.Series, bench_ret: pd.Series, ann_factor: int = 12) -> dict:
    """
    以最小平方法近似：r_s = alpha + beta * r_b + ε
    回傳：
      - beta
      - alpha_m : 月度 α（均值殘差）
      - alpha_a : 年化 α（(1+alpha_m)^A - 1）
      - corr    : 月度相關係數
    """
    s, b = align_series(strategy_ret, bench_ret)
    if len(s) < 3:
        return {"beta": np.nan, "alpha_m": np.nan, "alpha_a": np.nan, "corr": np.nan}

    cov = np.cov(s, b, ddof=1)[0, 1]
    var_b = np.var(b, ddof=1)
    beta = np.nan if var_b == 0 else cov / var_b

    # alpha_m 以期望值方式：E[s - beta*b]
    alpha_m = float((s - beta * b).mean())
    alpha_a = (1 + alpha_m) ** ann_factor - 1 if pd.notna(alpha_m) else np.nan

    corr = float(np.corrcoef(s, b)[0, 1])
    return {"beta": beta, "alpha_m": alpha_m, "alpha_a": alpha_a, "corr": corr}

# ------------------------------
# 風險/相對風險總表
# ------------------------------
def risk_summary(strategy_ret: pd.Series,
                 bench_ret: pd.Series,
                 ann_factor: int = 12) -> pd.DataFrame:
    """
    產出下列指標（策略、基準、超額）：
      - 年化報酬 AnnRet
      - 年化波動 AnnVol
      - Sharpe（RF=0）
      - 最大回撤 MDD
      - Calmar（AnnRet / |MDD|）
      - Beta、Corr（策略相對基準）
      - Tracking Error（TE）：策略-基準 的年化波動
      - Information Ratio（IR）：年化超額報酬 / TE
    """
    s, b = align_series(strategy_ret, bench_ret)
    if len(s) == 0:
        raise ValueError("risk_summary：沒有重疊的月份。")

    # 年化報酬（CAGR）
    def _ann_ret(r: pd.Series) -> float:
        if len(r) == 0:
            return np.nan
        cum = (1 + r).cumprod()
        return cum.iloc[-1] ** (ann_factor / len(r)) - 1.0

    # 年化波動
    def _ann_vol(r: pd.Series) -> float:
        return r.std() * np.sqrt(ann_factor)

    # Sharpe
    def _sharpe(r: pd.Series) -> float:
        vol = _ann_vol(r)
        if vol == 0:
            return np.nan
        return _ann_ret(r) / vol

    # 指標：策略/基準
    ann_s, ann_b = _ann_ret(s), _ann_ret(b)
    vol_s, vol_b = _ann_vol(s), _ann_vol(b)
    shr_s, shr_b = _sharpe(s), _sharpe(b)
    mdd_s, mdd_b = max_drawdown(s), max_drawdown(b)
    cal_s = np.nan if pd.isna(mdd_s) or mdd_s == 0 else ann_s / abs(mdd_s)
    cal_b = np.nan if pd.isna(mdd_b) or mdd_b == 0 else ann_b / abs(mdd_b)

    # 相對基準：Beta / Corr
    ba = calc_beta_alpha(s, b, ann_factor=ann_factor)
    beta, corr = ba["beta"], ba["corr"]

    # 超額序列
    diff = (s - b).dropna()
    te = diff.std() * np.sqrt(ann_factor) if len(diff) > 1 else np.nan
    er = _ann_ret(diff)  # 年化超額報酬（以差序列 CAGR）
    ir = (er / te) if (te and te != 0 and pd.notna(te)) else np.nan

    out = pd.DataFrame({
        "AnnRet":  [ann_s, ann_b, er],
        "AnnVol":  [vol_s, vol_b, te],
        "Sharpe":  [shr_s, shr_b, ir],
        "MDD":     [mdd_s, mdd_b, np.nan],
        "Calmar":  [cal_s, cal_b, np.nan],
        "Beta":    [beta, np.nan, np.nan],
        "Corr":    [corr, np.nan, np.nan],
    }, index=["Strategy", "Benchmark", "Excess( S - B )"])
    return out

# ------------------------------
# 滾動 Beta
# ------------------------------
def rolling_beta(strategy_ret: pd.Series,
                 bench_ret: pd.Series,
                 window: int = 24) -> pd.Series:
    """
    以移動視窗估計 Beta；window 預設 24 個月。
    不用 apply，直接用 rolling 協方差/變異數，穩定且快。
    """
    s, b = align_series(strategy_ret, bench_ret)
    if len(s) < window:
        return pd.Series(dtype=float)

    cov_sb = s.rolling(window).cov(b)      # Cov(s, b) 於每個月
    var_b  = b.rolling(window).var()       # Var(b)
    beta   = cov_sb / var_b
    beta.name = "rolling_beta"
    return beta

# ------------------------------
# 表格格式化（友善印出）
# ------------------------------
def format_percent_table(df: pd.DataFrame) -> str:
    """
    將 AnnRet/AnnVol/MDD/Calmar 以百分比格式顯示，Sharpe/IR/Beta/Corr 用小數。
    回傳：字串，適合直接 print。
    """
    df = df.copy()
    pct_cols = ["AnnRet", "AnnVol", "MDD", "Calmar"]
    for c in pct_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
    num_cols = ["Sharpe", "Beta", "Corr"]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    # Excess 列的 AnnVol=TE, Sharpe=IR，也沿用上面格式（已覆蓋）
    return df.to_string()

# ------------------------------
# 使用範例（直接跑這個檔案時）
# ------------------------------
if __name__ == "__main__":
    # 這段只是示意：假設你從 utils.run_backtest() 得到 res 物件
    # from utils import run_backtest
    # res = run_backtest(start="2015-01-01", benchmark="SPY")
    # s = res["bt"]["NetRet"]                # 策略淨報酬（月）
    # b = res["benchmark_aligned"]           # 基準月報酬
    # tab = risk_summary(s, b)
    # print(format_percent_table(tab))
    print("metrics.py ready. Import and use risk_summary/rolling_beta 等函數。")