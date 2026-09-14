"""候选池 / 可交易域的落地层（SQLite）。

设计依据：DESIGN §4.1（universe 表，每日更新并保留历史以便复盘）、
§4.5（candidate_pool，含 target_weight 与 factor_snapshot）。

铁律：本模块只写本地 SQLite，**不提交任何个人持仓/资金**（.gitignore 已挡）；
`positions` 按 §5.1 走 `positions.json`（手工维护），不进这里。
"""

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS universe (
  date TEXT NOT NULL, symbol TEXT NOT NULL, name TEXT,
  price REAL, change_pct REAL, amount_wan REAL, turnover_pct REAL,
  mktcap_yi REAL, industry TEXT, list_date TEXT, source TEXT,
  PRIMARY KEY (date, symbol)
);

CREATE TABLE IF NOT EXISTS candidate_pool (
  date TEXT NOT NULL, symbol TEXT NOT NULL, name TEXT,
  alpha REAL,                 -- 原始打分
  target_weight REAL,         -- 组合优化给出的目标权重
  factor_snapshot TEXT,       -- 当日因子值 JSON（复盘用）
  ml_score REAL,
  rank INTEGER,
  reason TEXT,                -- 可读理由（哪几个因子贡献大）
  PRIMARY KEY (date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_universe_date ON universe(date);
CREATE INDEX IF NOT EXISTS idx_pool_date ON candidate_pool(date);
"""


def connect(db_path: "str | Path") -> sqlite3.Connection:
    """打开（必要时创建）库并建表。WAL 便于读写并发（盘中轨会读）。"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def _upsert(conn: sqlite3.Connection, table: str, rows: list, cols: list) -> int:
    if not rows:
        return 0
    placeholders = ",".join("?" * len(cols))
    sql = ("INSERT INTO %s (%s) VALUES (%s) "
           "ON CONFLICT(date, symbol) DO UPDATE SET %s" % (
               table, ",".join(cols), placeholders,
               ",".join("%s=excluded.%s" % (c, c) for c in cols if c not in ("date", "symbol"))))
    conn.executemany(sql, [tuple(r.get(c) for c in cols) for r in rows])
    conn.commit()
    return len(rows)


def write_universe(conn, date: str, rows: list) -> int:
    """落可交易域。行字段来自 get_a_trade_universe 的 universe[]（同名对齐）。"""
    cols = ["date", "symbol", "name", "price", "change_pct", "amount_wan",
            "turnover_pct", "mktcap_yi", "industry", "list_date", "source"]
    return _upsert(conn, "universe", [dict(r, date=date) for r in rows], cols)


def write_candidates(conn, date: str, rows: list) -> int:
    """落候选池。factor_snapshot 允许传 dict（自动转 JSON），便于复盘回读。"""
    cols = ["date", "symbol", "name", "alpha", "target_weight",
            "factor_snapshot", "ml_score", "rank", "reason"]
    norm = []
    for r in rows:
        d = dict(r, date=date)
        if isinstance(d.get("factor_snapshot"), (dict, list)):
            d["factor_snapshot"] = json.dumps(d["factor_snapshot"], ensure_ascii=False)
        norm.append(d)
    return _upsert(conn, "candidate_pool", norm, cols)


def read_candidates(conn, date: str) -> list:
    """读某日候选池（按目标权重降序）——执行层与周报都从这里读。"""
    cur = conn.execute(
        "SELECT * FROM candidate_pool WHERE date=? ORDER BY target_weight DESC", (date,))
    return [dict(r) for r in cur.fetchall()]
