from utils import run_backtest, show_recent_decisions, DEFAULT_TICKERS
from metrics import risk_summary, format_percent_table, rolling_beta, calc_beta_alpha
from compare import compare_three

# res = run_backtest(
#     start="2010-01-01",
#     benchmark="SPY",
#     rebalance_freq="Q",     # 季度再平衡
#     rank_buffer=2,          # TopN+2 內續抱
#     min_trade=0.01,         # 小於 1% 的調整不交易
#     partial_rebalance=0.5   # 只移動一半
# )
# show_recent_decisions(res, n=6)

# 1. 回測

# 月調整 + 等權（基準線）
# res1 = run_backtest(benchmark="equal_weight", rebalance_freq="M")

# # 季調整 + 等權（先把換手壓下來）
# res2 = run_backtest(benchmark="equal_weight", rebalance_freq="Q")

# 季調整 + 波動率反比（風險效率版）
# res3 = run_backtest(benchmark="vol_scaled", rebalance_freq="Q")

# # 雙月調整 + 波動率反比（折衷）
# res = run_backtest(benchmark="vol_scaled", rebalance_freq="2M")

# 2. Beta and alpha

# 先跑策略
res = run_backtest(start="2010-01-01",top_n = 3, rebalance_freq = 'M', benchmark="SPY")
show_recent_decisions(res, n=6)
# 取策略報酬 & 基準報酬
s = res["bt"]["NetRet"]
b = res["benchmark_aligned"]

# 總表
tab = risk_summary(s, b)
print(format_percent_table(tab))
ba = calc_beta_alpha(s, b)
print(f"Beta={ba['beta']:.2f}, 月α={ba['alpha_m']:.2%}, 年化α={ba['alpha_a']:.2%}")

# 畫 rolling beta
rb = rolling_beta(s, b, window=24)
rb.plot(title="Rolling 24M Beta vs SPY")

# 3. 自我比較
# cfg1 = dict(start="2015-01-01", top_n=3, benchmark="SPY", rebalance_freq="2M")
# cfg2 = dict(start="2015-01-01", top_n=3, benchmark="SPY", rebalance_freq="M")
# cfg3 = dict(start="2015-01-01", top_n=3, benchmark="SPY", rebalance_freq="Q")

# compare_three(cfg1, cfg2, cfg3, label1="cfg1", label2="cfg2", label3="cfg3")