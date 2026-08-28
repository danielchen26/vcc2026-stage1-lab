"""vcclab — Arc Institute Virtual Cell Challenge 2026 Stage 1 实验库.

两级方法:
  Stage 1 (预测) : 哪些基因响应 / 涨跌 / 幅度  —— detectability / shrinkage / probit
  Stage 2 (构造) : 把预测无损翻译成 400 个整数细胞 —— decoder (已验证)
打分器复刻      : scorer.ControlRef
"""

from __future__ import annotations

from .decoder import design_cells, hamilton
from .detectability import mde
from .probit import gate_needs_nonparametric, p_detect
from .scorer import ALPHA, EPS, GATE_CPM, TS_BULK, TS_CELL, ControlRef, bh_adjust
from .shrinkage import james_stein, ledoit_wolf_nu0, tail_aware_scale

__version__ = "0.1.0"

__all__ = [
    "ALPHA",
    "EPS",
    "GATE_CPM",
    "TS_BULK",
    "TS_CELL",
    "ControlRef",
    "bh_adjust",
    "design_cells",
    "hamilton",
    "mde",
    "p_detect",
    "gate_needs_nonparametric",
    "james_stein",
    "ledoit_wolf_nu0",
    "tail_aware_scale",
]
