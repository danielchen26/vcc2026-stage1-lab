# E01 — RESULT

**原始结果获得于 2026-08-27** (本目录 `run.py` 是对当时探针代码的整理与固化).
机器: Apple M1 Pro / 16 GB / 无 CUDA (arm64 Darwin).
版本: cell-eval2 0.16.0 (本实验不调用), numpy 2.5.2, scipy 1.18.1, Python 3.12.0.

## 2026-08-27 原始输出

```
null draw    U_scipy=   3862696.0  U_psi=   3862696.0  match=True
shifted      U_scipy=   4735458.5  U_psi=   4735458.5  match=True
degenerate   U_scipy=   4771000.0  U_psi=   4771000.0  match=True
```

3/3 精确相等, 含点质量退化情形.

## 2026-08-28 复跑 (`run.py --real`)

```
=== E01 psi 闭式恒等 ===
日期=2026-08-28  机器=arm64 Darwin 3.12.0  numpy=2.5.2 scipy=1.18.1
数据=/Users/chetianc/vcc2026  基线=vcc_local.py

null draw    U_scipy=   3862696.0  U_psi=   3862696.0  match=True
shifted      U_scipy=   4735458.5  U_psi=   4735458.5  match=True
degenerate   U_scipy=   4771000.0  U_psi=   4771000.0  match=True

3/3 逐位相等

[真实数据闭环] ControlRef 加载 61.6s  gate=9929
  gate j=6322  零 10361/18400  U_scipy=   3938825.0  U_psi=   3938825.0  match=True
  gate j=5074  零  6494/18400  U_scipy=   4190717.5  U_psi=   4190717.5  match=True
  gate j=2678  零  3976/18400  U_scipy=   4399752.0  U_psi=   4399752.0  match=True
  gate j=3056  零  7660/18400  U_scipy=   4029476.5  U_psi=   4029476.5  match=True
  gate j=8442  零  5040/18400  U_scipy=   4237151.0  U_psi=   4237151.0  match=True
```

**与 2026-08-27 逐位一致** (三个 U 值一字不差; 固定种子 `default_rng(0)`, 无随机差异).
真实数据闭环 5/5 相等, 覆盖 3,976–10,361 个精确零的稀疏列 —— 零值快路径
`out[v == 0] = 0.5 * nzero` 与并列拆分都正确.

加载耗时 61.6 s (原始记录 12.8 s): 复跑时机器上另有两个并发 agent 在读同样的
h5ad, I/O 争用所致; 与数值结论无关.
