"""E31 构建器：把 build_v8.py / build_v9.py 的 build() 跑在 **E31 自己的 out 目录**里。

为什么要加这层：build_v8.py 的 `OUT` 指向 E28-pds/out，而 E28 归其他 agent 共用，
不能写。这里只重定向 `mod.OUT`（纯路径，不动任何数值逻辑），再把产物改名成
pred_<tag>.h5ad —— 改名走 Path.rename，同盘零拷贝，不占额外磁盘。

用法：
    python build_run.py v8     # 复跑 V8（force_mean 默认 True）—— 默认路径未变的对照
    python build_run.py v9     # V9（force_mean=False）
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SRC = {"v8": HERE.parents[0] / "E28-pds" / "build_v8.py", "v9": HERE / "build_v9.py"}


def load(path: Path):
    spec = importlib.util.spec_from_file_location(f"_e31_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    tag = sys.argv[1]
    free = shutil.disk_usage("/System/Volumes/Data").free / 1e9
    print(f"可用磁盘 {free:.1f} GB")
    if free < 5.0:
        raise SystemExit("磁盘不足 5 GB，按约定停止而不是继续")

    OUT.mkdir(parents=True, exist_ok=True)
    mod = load(SRC[tag])
    mod.OUT = OUT                      # 唯一的重定向：路径，不是逻辑
    mod.build()

    raw = OUT / "pred_v8.h5ad"         # build() 里的 tag 是字面量，不动它
    dst = OUT / f"pred_{tag}.h5ad"
    if raw != dst:
        raw.rename(dst)
    print(f"产物 {dst}  {dst.stat().st_size / 1e6:.0f} MB")
    (OUT / "_ctrl.h5ad").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
