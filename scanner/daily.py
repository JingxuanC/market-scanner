"""20:10 日频轨：可交易域 → h5 截面 → 粗筛 → ML 打分 → 组合优化 → candidate_pool。

设计依据：DESIGN §4.1（可交易域）/§4.4（组合优化出**权重**而不是 TopN 等权）/§4.5（落库）。

三个刻意的工程决定，都写在这里以便复审：
1. **as-of 截面必须完整**：修复期 h5 最新日可能只有 1091 行（全市场约 5200），
   拿它选股等于把 4/5 的票排除在外。所以取「最近一个截面行数 ≥ min_symbols 的交易日」，
   而不是无脑用 `max(datetime)`。
2. **粗筛用 20 日反转，定义与 hub 的 reversal20 完全一致**（`-close.pct_change(20)`）。
   之所以本地算而不是读 Redis `dfactor:`：修复期那批键只覆盖按代码序的前 1091 只
   （有偏子集），拿它当全市场筛子会系统性偏向 000xxx/600xxx。全量落库后可直接切回
   `dfactor:`（本模块留了 `alpha_source` 开关）。
3. **ML 缺失要显式记账**，不静默降级：ml_predict 失败时 ml_score 置空、reason 里写明
   `ml_unavailable`、summary.warnings 里列出。
"""

import json
import logging
import time
from pathlib import Path

from scanner import hub, store

log = logging.getLogger("scanner.daily")


def read_h5(h5_path) -> "object":
    import pandas as pd

    return pd.read_hdf(str(h5_path), key="data")


def pick_as_of(df, min_symbols: int = 3000):
    """最近一个「截面完整」的交易日（截面行数 ≥ min_symbols）。"""
    sizes = df.groupby(level="datetime").size()
    ok = sizes[sizes >= int(min_symbols)]
    if ok.empty:
        raise ValueError("没有截面行数 ≥ %d 的交易日（h5 可能损坏）" % min_symbols)
    return ok.index[-1]


def close_frame(df, as_of, lookback: int = 120):
    """as_of 往回 lookback 个交易日的收盘价宽表（行=日期，列=instrument 大写）。"""
    import pandas as pd

    dates = df.index.get_level_values("datetime")
    start = as_of - pd.Timedelta(days=int(lookback * 1.9) + 40)
    sub = df[(dates > start) & (dates <= as_of)]
    px = sub["$close"].unstack("instrument")
    return px.dropna(how="all").tail(int(lookback) + 1)


def bars_of(sub, symbol: str) -> list:
    """把 h5 的某只票切成 hub 要的 bar 列表 [{date,open,high,low,close,volume}, ...]。"""
    try:
        one = sub.xs(symbol, level="instrument")
    except KeyError:
        return []
    out = []
    for d, r in one.iterrows():
        c = r.get("$close")
        if c != c or c is None:  # NaN
            continue
        out.append({"date": d.strftime("%Y-%m-%d"), "open": float(r.get("$open", c)),
                    "high": float(r.get("$high", c)), "low": float(r.get("$low", c)),
                    "close": float(c), "volume": float(r.get("$volume", 0) or 0)})
    return out


def run(h5_path=None, db_path=None, as_of=None, shortlist: int = 200, top_n: int = 40,
        lookback: int = 120, min_symbols: int = 3000, min_amount_wan: float = 5000.0,
        use_ml: bool = True, method: str = "hrp", hub_mod=hub, alpha_source: str = "local",
        ml_max_symbols: int = 120, df=None, min_universe: int = 500) -> dict:
    """跑一次日频轨，返回 summary（并落 candidate_pool）。"""
    t0 = time.time()
    warn = []

    # 1) 可交易域（hub）
    uni = hub_mod.trade_universe(min_amount_wan=min_amount_wan)
    if isinstance(uni, dict) and uni.get("error"):
        raise hub.HubError("可交易域获取失败: %s" % uni)
    uni_rows = uni.get("universe") or []
    if len(uni_rows) < int(min_universe):
        raise hub.HubError("可交易域只有 %d 只 (< %d)，判定数据源退化，拒绝用它选股"
                           % (len(uni_rows), min_universe))
    by_sym = {r["symbol"].upper(): r for r in uni_rows}

    # 2) h5 + as-of 截面（df 可注入：测试不必依赖 pytables，服务器上也能只读一次复用）
    if df is None:
        if h5_path is None:
            raise ValueError("必须给 h5_path 或 df")
        df = read_h5(h5_path)
    as_of = as_of or pick_as_of(df, min_symbols)
    px = close_frame(df, as_of, lookback)
    dates = df.index.get_level_values("datetime")
    sub = df[(dates > (as_of - __import__("pandas").Timedelta(days=int(lookback * 1.9) + 40)))
             & (dates <= as_of)]
    cols = [c for c in px.columns if c in by_sym]
    if len(cols) < 2:
        raise ValueError("交集为空：可交易域与 h5 instrument 名对不上")
    px = px[cols]
    last = px.iloc[-1]
    prev = px.iloc[-21] if len(px) > 21 else px.iloc[0]
    alpha = -(last / prev - 1.0)                        # 20 日反转，定义同 hub reversal20
    alpha = alpha.dropna().sort_values(ascending=False)

    # 3) 粗筛
    short = list(alpha.index[:int(shortlist)])

    # 4) ML 打分（缺失显式记账）
    ml_scores = {}
    if use_ml and short:
        cand = short[:int(ml_max_symbols)]
        kl = [{"symbol": s, "klines": bars_of(sub, s)} for s in cand]
        kl = [x for x in kl if len(x["klines"]) >= 62]
        if kl:
            try:
                pred = hub_mod.ml_predict(kl)
                if isinstance(pred, dict):
                    # hub 的错误形状是 {"status":"error","message":...}，不是 {"error":...}；
                    # 而且它曾返回过 {"status":"ok","n_predicted":0}——**一只都没算却报成功**。
                    # 两种都必须显式报出来，绝不静默当"ML 没分"。
                    if pred.get("status") == "error":
                        raise hub.HubError(pred.get("message") or str(pred))
                    if pred.get("error"):
                        raise hub.HubError(str(pred["error"]))
                    if pred.get("n_predicted") == 0:
                        raise hub.HubError("ml_predict 返回 ok 但 n_predicted=0（等于没打分）")
                for s, v in (pred.get("predictions") or pred.get("scores") or {}).items():
                    ml_scores[str(s).upper()] = float(v)
                if not ml_scores:
                    warn.append("ml_predict 返回体里没有 predictions/scores 字段")
            except Exception as e:  # noqa: BLE001 — 显式记账，不静默
                warn.append("ml_predict 不可用: %s" % e)
        else:
            warn.append("ML 输入不足：没有 >=62 根 bar 的标的")
    elif not use_ml:
        warn.append("use_ml=False（本轮未接 ML 打分）")

    # 5) 排序取 top_n，再交给 hub 做组合优化（出权重，不是等权）
    ranked = sorted(short, key=lambda s: (-ml_scores.get(s, float("-inf")), -float(alpha[s])))
    pick = ranked[:int(top_n)]
    klines = {s: bars_of(sub, s) for s in pick}
    klines = {s: v for s, v in klines.items() if len(v) >= 21}
    pick = [s for s in pick if s in klines]
    if len(pick) < 2:
        raise ValueError("组合优化至少需要 2 只有效标的，实际 %d" % len(pick))
    opt = hub_mod.portfolio_optimize(pick, klines, method=method, lookback=lookback)
    if isinstance(opt, dict) and opt.get("error"):
        raise hub.HubError("组合优化失败: %s" % opt["error"])
    weights = opt.get("weights") or opt.get("target_weights") or opt
    weights = {str(k).upper(): float(v) for k, v in weights.items()
               if isinstance(v, (int, float))}

    # 6) 落库
    date = as_of.strftime("%Y-%m-%d")
    rows = []
    for i, s in enumerate(sorted(pick, key=lambda x: -weights.get(x, 0.0)), start=1):
        info = by_sym.get(s, {})
        mlv = ml_scores.get(s)
        reason = "反转20 前%d/%d" % (i, len(pick))
        reason += " | %s %.1f%%" % (method.upper(), 100 * weights.get(s, 0.0))
        reason += " | ml " + ("%.4f" % mlv if mlv is not None else "unavailable")
        rows.append({"symbol": s.lower(), "name": info.get("name"), "alpha": float(alpha.get(s, 0)),
                     "target_weight": weights.get(s, 0.0), "ml_score": mlv, "rank": i,
                     "reason": reason,
                     "factor_snapshot": {"reversal20": float(alpha.get(s, 0)),
                                         "amount_wan": info.get("amount_wan"),
                                         "industry": info.get("industry")}})
    conn = store.connect(db_path)
    n = store.write_candidates(conn, date, rows)
    conn.close()
    return {"date": date, "as_of": date, "universe": len(uni_rows),
            "universe_source": uni.get("source"), "shortlist": len(short),
            "picked": len(pick), "written": n, "weights_sum": round(sum(weights.values()), 4),
            "ml_scored": len(ml_scores), "warnings": warn,
            "elapsed_sec": round(time.time() - t0, 1)}
