"""实验台账的公共引导代码.

每个 `run.py` 的第一件事是 `from _common import ...`, 它做三件事:

1. 把 **可信基线** `~/vcc2026/vcc_local.py` 所在目录放进 `sys.path`.
   台账里的实验一律 pin 这个基线, 而不是 pin `src/vcclab/`:
   基线是数值行为的 oracle (它的 docstring 记录了对官方 cell-eval2 的
   逐基因验证), 移植后的包必须与它逐位一致才算移植成功.
   → 实验结果因此永远可以用来判定 `vcclab` 有没有回归.
2. 定位挑战数据 (context_*.h5ad / gene_names.csv / pert_counts.csv).
   数据绝不入库; 用环境变量 `VCC_DATA` 覆盖默认路径 `~/vcc2026`.
3. 打印一致的运行头 (日期 / 机器 / 版本), 让 stdout 可以直接贴进 RESULT.md.

用法::

    from _common import DATA, gene_names, header, load_ref
"""

from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

DATA = Path(os.environ.get("VCC_DATA", "~/vcc2026")).expanduser()
if not DATA.is_dir():
    raise SystemExit(
        f"找不到挑战数据目录 {DATA}. 用 VCC_DATA=/path/to/data 指定."
    )
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))          # 可信基线 vcc_local.py 就在这里


def gene_names():
    """18,533 个基因名, 顺序 = 官方 h5ad 的 var 顺序 (ControlRef 会校验)."""
    import pandas as pd

    return pd.read_csv(DATA / "gene_names.csv")["gene_name"].to_numpy()


def pert_list():
    """300 个待预测的扰动靶基因."""
    import pandas as pd

    return pd.read_csv(DATA / "pert_counts.csv")["target_gene"].to_numpy()


def load_ref(context: str = "A"):
    """加载一个 context 的 ControlRef, 返回 (ref, 耗时秒). 约 13 s / 1.4 GB."""
    from vcc_local import ControlRef

    t0 = time.time()
    ref = ControlRef(str(DATA / f"context_{context}.h5ad"), gene_names())
    return ref, time.time() - t0


def header(title: str) -> None:
    """把环境写进 stdout, 使输出可直接归档为 RESULT.md."""
    import numpy, scipy

    print(f"=== {title} ===")
    print(
        f"日期={time.strftime('%Y-%m-%d')}  机器={platform.machine()} "
        f"{platform.system()} {platform.python_version()}  "
        f"numpy={numpy.__version__} scipy={scipy.__version__}"
    )
    print(f"数据={DATA}  基线=vcc_local.py")
    print()
