"""服务器 bootstrap：跑一次日频轨。

为什么有个适配器：本轮 universe/ML 走 hub MCP，但 `portfolio_optimize` 直调容器内的
`analytics` —— 因为 hub 的服务进程还持有**修复前**的旧模块（docker cp 救不了已 import
进内存的模块），而重启容器会杀掉正在跑的全量日更抓取。
**容器重启后这个适配器就不需要了**，生产路径直接用 `scanner.hub.portfolio_optimize`。
"""
import json
import logging
import sys

sys.path.insert(0, "/app")   # 容器内 analytics.py 在 /app
sys.path.insert(0, "/tmp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

from scanner import daily, hub  # noqa: E402


class HubWithLocalOptimizer:
    def trade_universe(self, **kw):
        return hub.trade_universe(**kw)

    def ml_predict(self, kl, **kw):
        # 同上：走本地模块以拿到刚修的 trainer（服务进程还持有旧模块）
        import trainer  # noqa: PLC0415
        return trainer.get_trainer().predict(kl)

    def portfolio_optimize(self, symbols, klines, method="hrp", lookback=120, **kw):
        import analytics  # noqa: PLC0415 — 与 hub 服务暴露的是同一个函数
        return json.loads(analytics.portfolio_optimize(symbols, klines, method, lookback))


if __name__ == "__main__":
    out = daily.run(h5_path="/app/data/factor_mining/daily_pv_all.h5",
                    db_path="/app/usage/scanner.db",
                    shortlist=200, top_n=30, use_ml=True, ml_max_symbols=40,
                    min_amount_wan=5000.0, hub_mod=HubWithLocalOptimizer())
    print(json.dumps(out, ensure_ascii=False, indent=2))
