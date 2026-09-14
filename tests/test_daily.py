import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scanner import daily, hub, store  # noqa: E402

SYMS = ["SH600%03d" % i for i in range(1, 31)]


def make_df(days=140, incomplete_last=True):
    """30 只 × days 天；最后一天只有 2 只（模拟修复期截面不完整）。"""
    idx, close = [], []
    for s_i, sym in enumerate(SYMS):
        for d in range(days):
            idx.append((pd.Timestamp("2026-01-01") + pd.Timedelta(days=d), sym))
            close.append(10.0 + s_i * 0.5 + d * 0.01 * (1 if s_i % 2 else -1))
    if incomplete_last:
        for sym in SYMS[:2]:
            idx.append((pd.Timestamp("2026-01-01") + pd.Timedelta(days=days), sym))
            close.append(20.0)
    df = pd.DataFrame({"$close": close, "$open": close, "$high": close, "$low": close,
                       "$volume": [1000.0] * len(close)}, index=pd.MultiIndex.from_tuples(idx, names=["datetime", "instrument"]))
    return df


class FakeHub:
    def __init__(self, fail_ml=False):
        self.fail_ml = fail_ml

    def trade_universe(self, **kw):
        return {"source": "eastmoney_clist", "universe": [
            {"symbol": s.lower(), "name": "票" + s[-3:], "price": 10.0, "change_pct": 0.5,
             "amount_wan": 20000.0, "turnover_pct": 1.0, "mktcap_yi": 100.0,
             "industry": "测试", "list_date": "20200101"} for s in SYMS]}

    def ml_predict(self, klines_list, **kw):
        if self.fail_ml:
            raise hub.HubError("model missing")
        return {"status": "ok", "n_predicted": len(klines_list),
                "predictions": {k["symbol"]: 0.01 * i for i, k in enumerate(klines_list)}}

    def portfolio_optimize(self, symbols, klines, method="hrp", lookback=120, **kw):
        w = 1.0 / len(symbols)
        return {"method": method, "weights": {s: w for s in symbols}}


def test_pick_as_of_skips_incomplete_cross_section():
    df = make_df()
    got = daily.pick_as_of(df, min_symbols=10)
    assert got == pd.Timestamp("2026-01-01") + pd.Timedelta(days=139), "必须回退到完整截面"


def test_pick_as_of_raises_when_nothing_complete():
    df = make_df(days=3, incomplete_last=True)
    with pytest.raises(ValueError, match="没有截面行数"):
        daily.pick_as_of(df, min_symbols=100)


def test_run_end_to_end_writes_candidates(tmp_path):
    df = make_df()
    out = daily.run(db_path=tmp_path / "s.db", df=df, min_symbols=10, shortlist=20,
                    top_n=8, use_ml=True, hub_mod=FakeHub(), min_universe=10)
    assert out["as_of"] == "2026-05-20" and out["picked"] == 8 and out["written"] == 8
    assert out["weights_sum"] == pytest.approx(1.0, rel=1e-6)
    assert out["ml_scored"] == 20, "ML 打在粗筛 20 只上（不是最终 top_n）"
    rows = store.read_candidates(store.connect(tmp_path / "s.db"), out["date"])
    assert len(rows) == 8
    assert all(r["target_weight"] == pytest.approx(0.125) for r in rows)
    assert "HRP" in rows[0]["reason"] and "ml " in rows[0]["reason"]


def test_run_records_ml_failure_instead_of_silent(tmp_path):
    df = make_df()
    out = daily.run(db_path=tmp_path / "s.db", df=df, min_symbols=10, shortlist=20,
                    top_n=5, use_ml=True, hub_mod=FakeHub(fail_ml=True), min_universe=10)
    assert any("ml_predict 不可用" in w for w in out["warnings"]), "必须显式记账"
    rows = store.read_candidates(store.connect(tmp_path / "s.db"), out["date"])
    assert all("ml unavailable" in r["reason"] for r in rows)
    assert all(r["ml_score"] is None for r in rows)


def test_run_refuses_degenerate_universe(tmp_path):
    df = make_df()

    class Tiny(FakeHub):
        def trade_universe(self, **kw):
            return {"source": "x", "universe": [{"symbol": "sh600001"}]}

    with pytest.raises(hub.HubError, match="退化"):
        daily.run(db_path=tmp_path / "s.db", df=df, min_symbols=10, hub_mod=Tiny(), min_universe=10)


class SilentMlHub(FakeHub):
    """复现 hub 曾出现的形状：status=ok 但一只都没算。"""
    def ml_predict(self, klines_list, **kw):
        return {"status": "ok", "n_predicted": 0, "predictions": {}}


def test_run_treats_ok_but_zero_predicted_as_failure(tmp_path):
    df = make_df()
    out = daily.run(db_path=tmp_path / "s.db", df=df, min_symbols=10, shortlist=10,
                    top_n=3, use_ml=True, hub_mod=SilentMlHub(), min_universe=10)
    assert any("n_predicted=0" in w for w in out["warnings"]), "ok+0 必须被当成失败报出来"
    rows = store.read_candidates(store.connect(tmp_path / "s.db"), out["date"])
    assert all("ml unavailable" in r["reason"] for r in rows)
