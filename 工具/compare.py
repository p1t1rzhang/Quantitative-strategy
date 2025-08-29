# compare.py
import contextlib, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import run_backtest  # 你的 utils.py

# ------------------------------
# 小工具：靜音呼叫 run_backtest（方案 B）
#   - 暫時把 plt.show 換成 no-op
#   - 跑完關閉新產生的 figure，避免記憶體疊圖
#   - 同時把 stdout/stderr 靜音，避免 run_backtest 印表
# ------------------------------
def _silent_run_backtest(**kwargs):
    buf_out, buf_err = io.StringIO(), io.StringIO()
    old_show = plt.show
    existing_figs = set(plt.get_fignums())
    plt.show = lambda *a, **k: None  # 不彈圖
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            res = run_backtest(**kwargs)
    finally:
        # 關閉 run_backtest 期間新建的圖
        new_figs = set(plt.get_fignums()) - existing_figs
        for num in list(new_figs):
            try:
                plt.close(num)
            except Exception:
                pass
        plt.show = old_show  # 還原
    return res

def _is_res_obj(x):
    """判斷是否傳入的是 run_backtest 的結果物件"""
    return isinstance(x, dict) and {"bt","summary","benchmark_aligned"}.issubset(set(x.keys()))

def _extract_core(res):
    """
    從 run_backtest 結果取出核心資料：
    - net 月報酬、bench 月報酬（對齊）
    - 累積淨值、績效 summary、成本 table
    """
    bt   = res["bt"]
    net  = bt["NetRet"].copy()
    bench= res["benchmark_aligned"].copy()

    # 對齊日期
    common_idx = net.index.intersection(bench.index)
    net   = net.loc[common_idx]
    bench = bench.loc[common_idx]

    cum_net   = (1 + net).cumprod()
    cum_bench = (1 + bench).cumprod()

    return {
        "net": net,
        "bench": bench,
        "cum_net": cum_net,
        "cum_bench": cum_bench,
        "summary": res["summary"],
        "costs":   res["costs"]
    }

# ------------------------------
# 主函數：三策略比較（不做兩兩比較）
# ------------------------------
def compare_three(
    strat1, strat2, strat3,
    label1="策略1", label2="策略2", label3="策略3",
    plot_benchmark=True,
    plot_title="Cumulative Return (Net): 3 Strategies vs Benchmark",
    return_result=True
):
    """
    你可以傳兩種型態：
    1) 已跑好的結果物件（run_backtest 回傳的 dict）
       compare_three(res1, res2, res3, ...)
    2) 設定參數（dict），由本函數內部呼叫 run_backtest
       compare_three(cfg1, cfg2, cfg3, ...)，例如：
         cfg1 = dict(start="2015-01-01", top_n=3, benchmark="SPY", rebalance_freq="M")
    注意：三者會各自跑各自的設定；圖上只畫第一個策略的基準線。
    """

    # 取得三個結果物件（靜音版）
    res_list = []
    for x in (strat1, strat2, strat3):
        if _is_res_obj(x):
            res_list.append(x)
        elif isinstance(x, dict):
            res_list.append(_silent_run_backtest(**x))
        else:
            raise TypeError("strat 參數需為 run_backtest 結果物件或 kwargs 的 dict")

    core = [_extract_core(r) for r in res_list]

    # 對齊共同日期
    common_idx = core[0]["net"].index
    for i in [1,2]:
        common_idx = common_idx.intersection(core[i]["net"].index)

    net_series = [c["net"].loc[common_idx] for c in core]
    cum_series = [(1 + s).cumprod() for s in net_series]

    # 基準（只用第一個策略的）
    bench_cum = None
    if plot_benchmark:
        bench_line = core[0]["bench"].loc[common_idx]
        bench_cum = (1 + bench_line).cumprod()

    # ===== 圖：三策略＋（可選）基準 =====
    plt.figure(figsize=(11,6))
    plt.plot(cum_series[0], label=label1)
    plt.plot(cum_series[1], label=label2)
    plt.plot(cum_series[2], label=label3)
    if bench_cum is not None:
        plt.plot(bench_cum, label="Benchmark", alpha=0.75, linestyle="--")
    plt.title(plot_title)
    plt.grid(True); plt.legend()
    plt.xlabel("Date"); plt.ylabel("Cumulative Net Value")
    plt.tight_layout()
    plt.show()

    # ===== 績效總表（抓「策略（淨）」那一列）=====
    perf_rows = []
    for (c, label) in zip(core, [label1, label2, label3]):
        s = c["summary"]
        # 避免空白差異，用 startswith
        row = s.loc[s.index.str.startswith("策略（淨")]
        row.index = [label]
        perf_rows.append(row)
    perf_df = pd.concat(perf_rows, axis=0)
    perf_df = perf_df[["年化報酬率","年化波動率","夏普比率","最大回撤","Calmar比率"]]

    # ===== 換手/成本表 =====
    cost_rows = []
    for (c, label) in zip(core, [label1, label2, label3]):
        ct = c["costs"].copy()
        ct.index = [label]
        cost_rows.append(ct)
    cost_df = pd.concat(cost_rows, axis=0)

    # 美化輸出（百分比欄位轉格式字串）
    perf_fmt = perf_df.copy()
    for col in ["年化報酬率","年化波動率","最大回撤","Calmar比率"]:
        if col != "Calmar比率":
            perf_fmt[col] = perf_fmt[col].apply(lambda x: f"{x:.2%}")
    perf_fmt["夏普比率"] = perf_fmt["夏普比率"].apply(lambda x: f"{x:.2f}")

    cost_fmt = cost_df.copy()
    for col in ["年化換手率","年化成本拖累（毛-淨）","月均換手率","月均成本"]:
        cost_fmt[col] = cost_fmt[col].apply(lambda x: f"{x:.2%}")

    print("\n=== 績效指標（策略淨值，三者並排）===")
    print(perf_fmt.to_string())

    print("\n=== 換手與成本影響（三者並排）===")
    print(cost_fmt.to_string())

    if return_result:
        return {
            "index": common_idx,
            "cum": {
                label1: cum_series[0],
                label2: cum_series[1],
                label3: cum_series[2],
                "Benchmark": bench_cum
            },
            "perf_table": perf_df,
            "cost_table": cost_df
        }