# ===== 函數索引目錄 =====
# fetch_adj_close          : 下載 Yahoo Finance 價格資料（調整收盤價）
# calc_momentum_12_1       : 計算 12-1 動能
# build_equal_weight_topN  : 生成 TopN 等權配置
# backtest_with_costs      : 傳統回測引擎（含成本，純對照用）
# perf_stats               : 計算績效指標（年化報酬/波動/Sharpe/MDD/Calmar）
# _make_benchmark          : 處理基準（單一、等權、自訂組合）
# _make_rebalance_mask     : 產生再平衡月份索引（M/2M/Q）
# _sticky_topN_with_buffer : 動能選股（緩衝帶）
# run_backtest             : 一鍵回測主函數（含降換手、可換基準）
# show_recent_decisions    : 輔助輸出近期決策與動能排序
# ========================

# ===== 通用工具：12-1 動能 TopN 回測（含成本、可換基準、降換手選項） =====
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import contextlib, io

# ------------------------------
# 基本參數（可被 run_backtest 覆寫）
# ------------------------------
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "GLD"]
DEFAULT_START   = "2010-01-01"
DEFAULT_END     = None  # 最新
DEFAULT_TOP_N   = 3
DEFAULT_COST_BPS_ONE_WAY = 0.0005  # 5 bps
np.random.seed(42)

# ------------------------------
# 資料抓取 / 清理
# ------------------------------
def fetch_adj_close(tickers, start=DEFAULT_START, end=DEFAULT_END) -> pd.DataFrame:
    """下載 Yahoo Finance 資料並處理成調整收盤價"""
    raw = yf.download(
        tickers=tickers, start=start, end=end,
        auto_adjust=True, progress=False, group_by="ticker",
        threads=True
    )
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(-1):
            px = raw.xs("Close", axis=1, level=-1)
        else:
            px = raw.xs("Adj Close", axis=1, level=-1)
    else:
        if "Close" in raw.columns:
            px = raw[["Close"]].copy()
            colname = tickers[0] if len(tickers) == 1 else "Close"
            px.columns = [colname]
        else:
            px = raw[["Adj Close"]].copy()
            colname = tickers[0] if len(tickers) == 1 else "Adj Close"
            px.columns = [colname]

    cols = [c for c in tickers if c in px.columns]
    px = px[cols].sort_index().ffill()
    data_start = px.dropna(how="all").index.min()
    data_end   = px.dropna(how="all").index.max()
    print(f"下載資料期間：{data_start.date()} ~ {data_end.date()}")
    print(f"最後更新交易日（資料最末日）：{data_end.date()}")
    return px

# ------------------------------
# 訊號與權重（Jegadeesh & Titman, 1993）「12-1 動能」
# ------------------------------
def calc_momentum_12_1(monthly_px: pd.DataFrame) -> pd.DataFrame:
    """12-1 動能指標"""
    return monthly_px.shift(2) / monthly_px.shift(13) - 1.0

def build_equal_weight_topN(mom_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Top N 等權配置"""
    weights = pd.DataFrame(0.0, index=mom_df.index, columns=mom_df.columns)
    for dt, row in mom_df.iterrows():
        row = row.dropna()
        if row.empty: continue
        top = row.sort_values(ascending=False).head(top_n).index
        weights.loc[dt, top] = 1.0 / top_n
    return weights

# ------------------------------
# 權重計算函數（支持等權 / 波動率反比）
# ------------------------------
def build_weights(m_px, signals, top_n=3, method="equal"):
    """
    m_px     : 月度價格
    signals  : 動能分數 DataFrame
    top_n    : 選前 N 檔
    method   : "equal" / "vol_scaled"
    """
    w = pd.DataFrame(0, index=signals.index, columns=signals.columns)
    
    for dt in signals.index:
        top_assets = signals.loc[dt].nlargest(top_n).index
        if method == "equal":
            w.loc[dt, top_assets] = 1.0 / top_n
        elif method == "vol_scaled":
            vols = m_px[top_assets].pct_change().rolling(12).std().iloc[-1]  # 12m vol
            inv_vol = 1 / vols
            weights = inv_vol / inv_vol.sum()
            w.loc[dt, top_assets] = weights
    return w

# ------------------------------
# 週期下採樣（改持有頻率）
# ------------------------------
def downsample_weights(weights, freq="M"):
    """
    freq: "M"=月, "2M"=雙月, "Q"=季
    """
    return weights.resample(freq).first().reindex(weights.index, method="ffill")

# ------------------------------
# 經典版回測引擎（純粹對照用）
# ------------------------------
def backtest_with_costs(monthly_ret: pd.DataFrame,
                        target_weights: pd.DataFrame,
                        cost_rate: float = DEFAULT_COST_BPS_ONE_WAY) -> pd.DataFrame:
    """月頻回測（含成本）"""
    common_idx = target_weights.index.intersection(monthly_ret.index)
    target_weights = target_weights.loc[common_idx]
    monthly_ret    = monthly_ret.loc[common_idx]
    tickers = target_weights.columns
    records = []
    w_prev = pd.Series(0.0, index=tickers)
    for i in range(len(common_idx) - 1):
        t, t_next = common_idx[i], common_idx[i+1]
        r_t   = monthly_ret.loc[t, tickers].fillna(0.0)
        w_tgt = target_weights.loc[t].fillna(0.0)
        if i == 0:
            w_prev_after = w_prev.copy()
        else:
            port_ret_t = float((w_prev * r_t).sum())
            denom = (1.0 + port_ret_t) if (1.0 + port_ret_t) != 0 else 1.0
            w_prev_after = (w_prev * (1.0 + r_t)) / denom
        trades = w_tgt - w_prev_after
        turnover_t = float(np.abs(trades).sum())
        cost_t = turnover_t * cost_rate
        r_next = monthly_ret.loc[t_next, tickers].fillna(0.0)
        gross_ret_next = float((w_tgt * r_next).sum())
        net_ret_next   = gross_ret_next - cost_t
        records.append({
            "Month": t_next,"GrossRet": gross_ret_next,
            "NetRet": net_ret_next,"Turnover": turnover_t,"Cost": cost_t
        })
        w_prev = w_tgt.copy()
    return pd.DataFrame(records).set_index("Month")

# ------------------------------
# 績效指標
# ------------------------------
def perf_stats(returns: pd.Series, ann_factor: int = 12):
    """計算 AnnRet, AnnVol, Sharpe, MDD, Calmar"""
    r = returns.dropna()
    if r.empty:
        return {"AnnRet": np.nan,"AnnVol": np.nan,"Sharpe": np.nan,"MDD": np.nan,"Calmar": np.nan}
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    mdd = dd.min()
    n = len(r)
    ann_ret = cum.iloc[-1]**(ann_factor / n) - 1.0
    ann_vol = r.std() * np.sqrt(ann_factor)
    sharpe  = np.nan if ann_vol == 0 else ann_ret / ann_vol
    calmar  = np.nan if mdd == 0 else ann_ret / abs(mdd)
    return {"AnnRet": ann_ret,"AnnVol": ann_vol,"Sharpe": sharpe,"MDD": mdd,"Calmar": calmar}

# ------------------------------
# 基準：波動率反比（Inverse-Vol）月度權重（滾動）
# ------------------------------
def _make_vol_scaled_benchmark(m_ret: pd.DataFrame,
                               tickers_all: list[str],
                               window: int = 36,
                               min_assets: int = 3) -> pd.Series:
    """
    用過去 window 個月的標準差做 1/σ 權重（前視避免：std 用 shift(1)）。
    權重每月更新；若可用資產 < min_assets 則退回等權。
    回傳：基準月報酬序列（pd.Series）
    """
    # 限定宇宙欄位
    R = m_ret.reindex(columns=tickers_all)
    # 過去 window 月的波動（用 shift(1) 避免偷看當月）
    vol = R.rolling(window=window, min_periods=window).std().shift(1)

    inv_vol = 1.0 / vol
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)

    # 權重正規化
    w = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    # 資產數不足則等權
    valid_count = w.notna().sum(axis=1)
    ew = pd.DataFrame(index=w.index, columns=w.columns, data=0.0)
    ew.loc[:, :] = 1.0 / len(tickers_all)
    w = np.where((valid_count.values[:, None] >= min_assets), w, ew)
    w = pd.DataFrame(w, index=R.index, columns=R.columns)

    # 每月基準報酬
    bench = (w * R).sum(axis=1)
    return bench.dropna()

# ------------------------------
# 基準工具：把 benchmark 參數轉成月報酬序列
# ------------------------------
def _make_benchmark(m_ret: pd.DataFrame, benchmark, tickers_all) -> tuple[pd.Series, str]:
    """
    支援：
      1) str：
         - 單一 ticker，例如 "SPY"、"QQQ"
         - "equal_weight"：全宇宙等權
         - "vol_scaled"：波動率反比（滾動 36m 1/σ 權重）
      2) dict：加權字典，例如 {"SPY":0.5,"QQQ":0.5}（自動正規化）
    回傳：(benchmark_m_ret, label)
    """
    if isinstance(benchmark, str):
        b = benchmark.lower()
        if b == "equal_weight":
            bench = m_ret[tickers_all].mean(axis=1)
            label = "Equal-Weight (Universe)"
        elif b == "vol_scaled":
            bench = _make_vol_scaled_benchmark(m_ret, tickers_all, window=36, min_assets=3)
            label = "Vol-Scaled (36m Inverse-Vol)"
        else:
            # 單一資產
            if benchmark not in m_ret.columns:
                raise ValueError(f"benchmark='{benchmark}' 不在回測宇宙")
            bench = m_ret[benchmark].dropna()
            label = benchmark
    elif isinstance(benchmark, dict):
        keys = list(benchmark.keys())
        vals = np.array(list(benchmark.values()), dtype=float)
        if not np.isclose(vals.sum(), 1.0):
            vals = vals / vals.sum()
        for k in keys:
            if k not in m_ret.columns:
                raise ValueError(f"benchmark 權重包含未知標的：{k}")
        bench = (m_ret[keys] * vals).sum(axis=1).dropna()
        parts = [f"{k}:{w:.0%}" for k, w in zip(keys, vals)]
        label = "Custom[" + ",".join(parts) + "]"
    else:
        raise ValueError("benchmark 需為 str、dict、'equal_weight' 或 'vol_scaled'")
    return bench, label

# ------------------------------
# 降換手工具：排程與緩衝
# ------------------------------
def _make_rebalance_mask(idx: pd.DatetimeIndex, freq: str) -> pd.Index:
    """產生再平衡月份索引"""
    f = freq.upper()
    if f == "M": return idx
    elif f == "2M": return idx[::2]
    elif f == "Q": return idx[idx.month.isin([3,6,9,12])]
    else: raise ValueError("rebalance_freq 僅支援 'M','2M','Q'")

def _sticky_topN_with_buffer(rank_series: pd.Series,
                             current_hold: list[str],
                             top_n: int,
                             buffer: int) -> list[str]:
    """排名緩衝選股"""
    ranked = rank_series.sort_values(ascending=False).index.tolist()
    keep_cut = top_n + max(buffer, 0)
    keep = [t for t in current_hold if t in ranked[:keep_cut]]
    for t in ranked[:top_n]:
        if t not in keep and len(keep) < top_n:
            keep.append(t)
    return keep[:top_n]

# ------------------------------
# 一鍵回測（可換時段／宇宙／基準／降換手）
# ------------------------------
def run_backtest(start: str = DEFAULT_START,
                 end: str | None = DEFAULT_END,
                 tickers: list[str] = None,
                 top_n: int = DEFAULT_TOP_N,
                 cost_bps: float = DEFAULT_COST_BPS_ONE_WAY,
                 benchmark="SPY",
                 # ↓↓↓ 降換手參數
                 rebalance_freq: str = "Q",     # "M"=月, "2M"=雙月, "Q"=季（預設季頻）
                 rank_buffer: int = 2,          # 舊持倉掉到 TopN+buffer 仍續抱
                 min_trade: float = 0.01,       # 不交易帶：單資產調整<1%就不動
                 partial_rebalance: float = 0.5 # 部分再平衡：只移動 50%
                 ):
    """主要回測函數：12-1 動能 TopN 等權，含可換基準與降換手設定。"""
    if tickers is None:
        tickers = DEFAULT_TICKERS

    # 下載資料 & 月化
    px = fetch_adj_close(tickers, start, end)
    px = px[px.index >= pd.to_datetime(start)]
    m_px  = px.resample("M").last()
    m_ret = m_px.pct_change()

    # 基準
    bench_m_ret, bench_label = _make_benchmark(m_ret, benchmark, tickers)
    print(f"月資料起訖：{m_px.index.min().date()} → {m_px.index.max().date()}（基準：{bench_label}）")

    # 訊號（12-1）
    mom_12_1 = calc_momentum_12_1(m_px)

    # ====== 降換手版回測：排程 + 緩衝 + 不交易帶 + 部分再平衡 ======
    m_idx = m_ret.index
    rebalance_dates = _make_rebalance_mask(m_idx, rebalance_freq)
    tickers_list = list(m_ret.columns)

    records = []
    w_prev = pd.Series(0.0, index=tickers_list)  # 上期實際權重
    current_hold = []                            # 目前持倉清單
    holdings_rows = []                           # t 決策 → t+1 持有

    for i in range(len(m_idx) - 1):
        t, t_next = m_idx[i], m_idx[i + 1]
        r_t = m_ret.loc[t, tickers_list].fillna(0.0)

        # 市場漂移（t 月內）
        if i == 0:
            w_prev_after = w_prev.copy()
        else:
            port_ret_t = float((w_prev * r_t).sum())
            denom = (1.0 + port_ret_t) if (1.0 + port_ret_t) != 0 else 1.0
            w_prev_after = (w_prev * (1.0 + r_t)) / denom

        # 預設：不調整（維持漂移後權重）
        w_tgt = w_prev_after.copy()

        # 若 t 為再平衡月份 → 依動能做決策
        if t in rebalance_dates:
            scores = mom_12_1.loc[t].dropna().reindex(tickers_list).dropna()
            chosen = _sticky_topN_with_buffer(scores, current_hold, top_n=top_n, buffer=rank_buffer)

            # 紀錄「t+1 的持倉成分」
            top_list = chosen + [""] * max(0, top_n - len(chosen))
            holdings_rows.append([t_next.date()] + top_list[:top_n])

            # 目標等權（只在 chosen 上）
            w_tgt = pd.Series(0.0, index=tickers_list)
            if len(chosen) > 0:
                w_tgt.loc[chosen] = 1.0 / len(chosen)

            # 不交易帶（no-trade band）
            delta = w_tgt - w_prev_after
            small = delta.abs() < float(min_trade)
            delta[small] = 0.0
            w_tgt = w_prev_after + delta

            # 部分再平衡（partial rebalance）
            pr = float(partial_rebalance)
            if 0.0 < pr < 1.0:
                w_tgt = w_prev_after + pr * (w_tgt - w_prev_after)

            # 正規化（避免數值累積誤差）
            s = w_tgt.sum()
            w_tgt = w_tgt / s if s > 0 else pd.Series(0.0, index=tickers_list)

            # 更新目前持倉清單
            current_hold = chosen.copy()

        # 當期換手與成本（在 t 月底發生，影響 t+1 淨報酬）
        trades = w_tgt - w_prev_after
        turnover_t = float(np.abs(trades).sum())
        cost_t = turnover_t * float(cost_bps)

        # 以 w_tgt 承擔 t+1 報酬
        r_next = m_ret.loc[t_next, tickers_list].fillna(0.0)
        gross_ret_next = float((w_tgt * r_next).sum())
        net_ret_next   = gross_ret_next - cost_t

        records.append({
            "Month": t_next,
            "GrossRet": gross_ret_next,
            "NetRet": net_ret_next,
            "Turnover": turnover_t,
            "Cost": cost_t
        })

        # 進入下一期
        w_prev = w_tgt.copy()

    bt = pd.DataFrame(records).set_index("Month")
    # ====== 降換手版回測結束 ======

    # 成分表 DataFrame
    holdings_cols = ["持有月份"] + [f"Top{i+1}" for i in range(top_n)]
    holdings_df = pd.DataFrame(holdings_rows, columns=holdings_cols)

    # 對齊基準
    bt = bt.loc[bt.index.intersection(bench_m_ret.index)]
    bench_aligned = bench_m_ret.loc[bt.index]

    # 指標
    stats_net   = perf_stats(bt["NetRet"])
    stats_gross = perf_stats(bt["GrossRet"])
    stats_bench = perf_stats(bench_aligned)
    summary = pd.DataFrame({
        "年化報酬率":  [stats_net["AnnRet"],  stats_gross["AnnRet"],  stats_bench["AnnRet"]],
        "年化波動率":  [stats_net["AnnVol"],  stats_gross["AnnVol"],  stats_bench["AnnVol"]],
        "夏普比率":   [stats_net["Sharpe"],  stats_gross["Sharpe"],  stats_bench["Sharpe"]],
        "最大回撤":   [stats_net["MDD"],     stats_gross["MDD"],     stats_bench["MDD"]],
        "Calmar比率": [stats_net["Calmar"],  stats_gross["Calmar"],  stats_bench["Calmar"]],
    }, index=["策略（淨）","策略（毛）", f"基準（{bench_label}）"])

    # 成本與換手
    monthly_turnover = bt["Turnover"]
    annualized_turnover = monthly_turnover.mean() * 12
    ann_cost_drag = summary.loc["策略（毛）","年化報酬率"] - summary.loc["策略（淨）","年化報酬率"]
    costs_summary = pd.DataFrame({
        "年化換手率": [annualized_turnover],
        "年化成本拖累（毛-淨）": [ann_cost_drag],
        "月均換手率": [monthly_turnover.mean()],
        "月均成本": [bt["Cost"].mean()],
    })

    # 視覺化（英文；標籤帶入基準名）
    cum_gross = (1 + bt["GrossRet"]).cumprod()
    cum_net   = (1 + bt["NetRet"]).cumprod()
    cum_bench = (1 + bench_aligned).cumprod()

    plt.figure(figsize=(10,6))
    plt.plot(cum_net,   label="Strategy (Net)")
    plt.plot(cum_gross, label="Strategy (Gross)", alpha=0.6)
    plt.plot(cum_bench, label=f"Benchmark ({bench_label})")
    plt.title("Cumulative Return: Strategy vs Benchmark")
    plt.legend(); plt.grid(True); plt.show()

    plt.figure(figsize=(8,5))
    plt.hist(bt["NetRet"].dropna(), bins=30)
    plt.title("Distribution of Strategy (Monthly Net Returns)")
    plt.xlabel("Monthly Return")
    plt.ylabel("Frequency")
    plt.grid(True); plt.show()

    # 印表
    print("績效指標總表")
    print(summary.to_string(float_format=lambda x: f"{x:.2%}" if abs(x) < 10 else f"{x:.2f}"))
    print("\n換手與成本影響")
    print(costs_summary.to_string(float_format=lambda x: f"{x:.2%}"))

    return {
        "px": px, "m_px": m_px, "m_ret": m_ret,
        "bt": bt, "benchmark_aligned": bench_aligned,
        "summary": summary, "costs": costs_summary,
        "holdings": holdings_df, "mom": mom_12_1
    }

# ------------------------------
# 報表輔助
# ------------------------------
def show_recent_decisions(res: dict, n: int = 6):
    """列印最近 n 個決策月動能排序"""
    mom = res["mom"]
    sample_months = mom.dropna(how="all").index[-n:]
    for dt in sample_months:
        print(f"\n== 決策月份：{dt.date()}（下月生效）==")
        print("12-1 動能排序:")
        print(mom.loc[dt].sort_values(ascending=False).round(4).to_string())


