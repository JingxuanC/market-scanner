"""在服务器上跑一次日频轨（bootstrap 用：不经容器，直接复用 factor-miner 容器内的 pandas + h5）。"""
import json
import logging
import sys

sys.path.insert(0, "/tmp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

from scanner import daily  # noqa: E402

out = daily.run(h5_path="/app/data/factor_mining/daily_pv_all.h5",
                db_path="/app/usage/scanner.db",
                shortlist=200, top_n=30, use_ml=True, ml_max_symbols=40,
                min_amount_wan=5000.0)
print(json.dumps(out, ensure_ascii=False, indent=2))
