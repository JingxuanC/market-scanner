import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scanner import store  # noqa: E402


def test_schema_and_universe_upsert(tmp_path):
    conn = store.connect(tmp_path / "s.db")
    rows = [{"symbol": "sh600519", "name": "贵州茅台", "price": 1700.0, "change_pct": 1.0,
             "amount_wan": 1e6, "turnover_pct": 0.5, "mktcap_yi": 21000.0,
             "industry": "白酒", "list_date": "20010827"}]
    assert store.write_universe(conn, "2026-09-14", rows) == 1
    assert store.write_universe(conn, "2026-09-14", rows) == 1  # 幂等
    got = conn.execute("SELECT COUNT(*) c FROM universe WHERE date='2026-09-14'").fetchone()["c"]
    assert got == 1, "同 (date,symbol) 必须 upsert 而不是插两行"


def test_candidates_roundtrip_and_order(tmp_path):
    conn = store.connect(tmp_path / "s.db")
    store.write_candidates(conn, "2026-09-14", [
        {"symbol": "sh600519", "name": "茅台", "alpha": 0.9, "target_weight": 0.08,
         "factor_snapshot": {"reversal20": 0.011}, "ml_score": 0.7, "rank": 1, "reason": "反转+质量"},
        {"symbol": "sz000001", "name": "平安银行", "alpha": 0.4, "target_weight": 0.12,
         "factor_snapshot": {"reversal20": -0.01}, "ml_score": 0.3, "rank": 2, "reason": "低估值"},
    ])
    out = store.read_candidates(conn, "2026-09-14")
    assert [r["symbol"] for r in out] == ["sz000001", "sh600519"], "按 target_weight 降序"
    assert json.loads(out[1]["factor_snapshot"])["reversal20"] == 0.011


def test_upsert_updates_fields(tmp_path):
    conn = store.connect(tmp_path / "s.db")
    store.write_candidates(conn, "2026-09-14", [{"symbol": "sh600519", "target_weight": 0.05}])
    store.write_candidates(conn, "2026-09-14", [{"symbol": "sh600519", "target_weight": 0.09}])
    row = conn.execute("SELECT target_weight FROM candidate_pool").fetchone()
    assert row["target_weight"] == 0.09
