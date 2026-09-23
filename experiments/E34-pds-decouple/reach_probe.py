"""E34c · `direction_reach` 的**细胞级**本地复现器 —— 找出 V14 失败的第三个来源。

V14 实测（`out/score_v14.json`）：不变量 `max|off| < min|lfc_t|` 在 build 内断言通过，
但 `reach` raw 仍是 0.0208，**低于** T13 的零假设 0.0355。SPEC §8e 停止条件 1 触发：
「纯前缀还有第三个来源，§8a 仍不完整」。本文件的任务是把那个来源**测出来**。

## 为什么 `cap_probe.py` 看不见它（方法学，不是笔误）

`cap_probe.py:91` 走的是 `m._pred_bulk_closed(m_full, lf, perts, bts)` —— **闭式
pseudobulk，不生成细胞**。没有细胞就没有 Wilcoxon，就没有 `p_adj_pred`。而
`cell_eval2/metrics/direction.py:508-511` 的排序键是：

    .sort(["target", "_sig_pred", "rank_p_adj", "rank_p_value", "abs_lfc_pred", "feature"],
          descending=[False, True, False, False, True, False])

`abs_lfc_pred` 是**第 4 个键**，只在 p 值并列时才起作用；第 1 个键是「预测是否显著」。
所以闭式探针能看到的量（幅度）恰好是对排序**几乎不起作用**的那一个，它结构上不可能
判出 reach。压幅压的是第 4 个键 —— 这就是 §8a 的错处。

## 本文件复现的是官方量本身，不是代理量

- `scorer.py:112-114` 的 `psi_g(v) = #{ctrl<v} + 0.5#{ctrl==v}` 是 Wilcoxon 的充分统计量，
  注释写明「对 `scipy.mannwhitneyu` 逐位验证」；`scorer.py:138-160` 的 `de_table` 是
  「cell-eval2 wilcoxon DE 的精确复刻」。
- `competition.py:52` `CONTROL_SOURCE_SCORED = "real"` —— 打分器拿我们的细胞对**真实
  对照池**做检验，正是 `ControlRef` 持有的那 18,400 个 NTC 细胞。`log1p` 是单调变换，
  不改秩，故 psi 对官方 lognorm 检验也精确。
- real 侧直接读 `E27-six-metrics/out/real.h5ad` —— 每次官方打分用的**同一个冻结文件**。

判为可信的条件（`--validate`）：本地 reach 必须复现三个已知的官方值
V8 0.1482 / V13 0.0250 / V14 0.0208。复现不出来就不许用它筛设计。

跑法：
    P=/Users/chetianc/vcc2026/.venv/bin/python
    $P experiments/E34-pds-decouple/reach_probe.py validate
    $P experiments/E34-pds-decouple/reach_probe.py v15
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.decoder import design_cells  # noqa: E402
from vcclab.scorer import EPS, TS_CELL, ControlRef, bh_adjust  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_CTRL_CELLS, VCC_PERT_CELLS, VCC_UMI, read_rows, thin  # noqa: E402

DATA = ROOT / "data"
H5 = DATA / "vcc2025" / "adata_Validation.h5ad"
MAPCSV = DATA / "external" / "ens2sym.csv"
REAL = ROOT / "experiments" / "E27-six-metrics" / "out" / "real.h5ad"
OUT = Path(__file__).resolve().parent / "out"
CACHE = OUT / "_reach_cache.npz"

N_PERT, ALPHA, SEED = 8, 0.05, 0
LAMBDA, K_CALL = 0.7, 288
K_OFF, LAMBDA_OFF = 550, 1.0
CAP_MARGIN = 0.99
PURITY_FLOOR = 0.9          # cell_eval2/metrics/direction.py:29 REACH_PURITY_FLOOR

# 官方实测，用于判定本复现器是否可信（出处：out/score_v*.json, E28/agg_v8.parquet）
OFFICIAL = {"v8": 0.1482, "v13": 0.0250, "v14": 0.0208}


def as_str(g):
    if isinstance(g, h5py.Group) and "categories" in g:
        cat = np.array([x.decode() if isinstance(x, bytes) else str(x)
                        for x in g["categories"][:]])
        return cat[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


# ---------------------------------------------------------------- DE (p, p_adj, lfc)
def de_full(ref: ControlRef, counts: np.ndarray):
    """(p, p_adj, lfc)，gate 内。`de_table` 只返回 (p_adj, lfc)，而官方排序键第 3 位是
    `rank_p_value`（BH 平台内的细分），所以这里必须把**原始 p** 也拿出来。

    与 `de_table` 唯一的差别是并列项 T[j] 走可加分解而不是对 18,800 个值做 np.unique：
        T = T_ctrl + Σ_{v 中的每个不同值 x} [(c_ctrl(x)+c_v(x))³-(…)] - [c_ctrl(x)³-c_ctrl(x)]
    逐位等价（`validate` 里对 `de_table` 断言 allclose）。
    """
    n1, n2 = counts.shape[0], ref.n_ctrl
    N = n1 + n2
    U = np.empty(ref.G)
    T = np.empty(ref.G)
    for j in range(ref.G):
        v = counts[:, ref.gidx[j]].astype(np.float64)
        U[j] = ref.psi(j, v).sum()
        col = ref._sorted[j]
        nz = float(ref._nzero[j])
        base = ref.tie_cube_sum_ctrl(j)
        xs, cv = np.unique(v, return_counts=True)
        cc = np.where(xs == 0.0, nz,
                      np.searchsorted(col, xs, "right") - np.searchsorted(col, xs, "left"))
        tot = cc + cv
        T[j] = base + float(np.sum(tot**3 - tot) - np.sum(cc**3 - cc))
    sd = np.sqrt(n1 * n2 / 12.0 * ((N + 1) - T / (N * (N - 1.0))))
    p = 2 * norm.sf(np.abs((U - n1 * n2 / 2) / sd))
    lfc = np.log2((counts[:, ref.gidx].mean(0) + EPS) / (ref.m_gate + EPS))
    return p, bh_adjust(p), lfc


# ---------------------------------------------------------------- reach 复现
def reach_one(names, p_adj_r, lfc_r, p_pred, p_adj_p, lfc_p, target):
    """`de_wilcoxon_direction_reach_raw` 的单扰动值 k*/N_conf，逐条照
    `direction.py` 的 `_reference_stats` / `_purity_curve` / `_k_star`。

    返回 (reach, k_star, n_conf, head)，`head` 是前缀最前面 12 条的
    (基因, 是否显著, p_adj, |lfc|, 是否命中, 是否集外) —— 用来直接看谁占了头部。
    """
    keep = names != target                       # on-target 排除（两侧都排）
    nm, pr, lr = names[keep], p_adj_r[keep], lfc_r[keep]
    pp, pap, lp = p_pred[keep], p_adj_p[keep], lfc_p[keep]

    n_conf = int((pr < ALPHA).sum())
    if n_conf == 0:
        return np.nan, 0, 0, []

    pool = pr < ALPHA                            # universe='adjudicated'
    nm, lr, pp, pap, lp = nm[pool], lr[pool], pp[pool], pap[pool], lp[pool]

    in_denom = np.isfinite(lr) & (lr != 0.0)
    committed = np.isfinite(lp) & (lp != 0.0)
    match = in_denom & committed & (np.sign(lp) == np.sign(lr))

    sig = pap < ALPHA
    abs_lp = np.where(committed, np.abs(lp), -np.inf)
    # np.lexsort：最后一个键最主要。官方顺序 = sig desc, p_adj asc, p asc, |lfc| desc, name asc
    order = np.lexsort((nm, -abs_lp, pp, pap, ~sig))

    d = np.cumsum(in_denom[order])
    mt = np.cumsum(match[order])
    with np.errstate(invalid="ignore", divide="ignore"):
        purity = np.where(d > 0, mt / np.maximum(d, 1), np.nan)
    ok = (d > 0) & (purity >= PURITY_FLOOR)
    k_star = int(d[ok].max()) if ok.any() else 0
    head = [(nm[order][i], bool(sig[order][i]), float(pap[order][i]),
             float(abs_lp[order][i]), bool(match[order][i])) for i in range(min(12, nm.size))]
    return k_star / n_conf, k_star, n_conf, head


# ---------------------------------------------------------------- 管线
def load_pipeline():
    """ref / 8 个扰动 / 召集集 / 集外向量 —— 逐字照 `build_v14.py:118-226`。"""
    with h5py.File(H5, "r") as f:
        genes = as_str(f["var"]["_index"])
        tg_all = as_str(f["obs"]["target_gene"])
    ctrl_path = OUT / "_ctrl_probe.h5ad"
    OUT.mkdir(parents=True, exist_ok=True)

    def _write_ctrl():
        """对照 h5ad 逐字照 build_v14.py:126-134 的 rng 顺序生成，写完即缓存复用。
        不每轮重建：磁盘紧张时 gzip 写一半会留下读不开的文件（本轮真的踩到）。"""
        import anndata as ad
        rng = np.random.default_rng(SEED)
        ntc_rows = rng.choice(np.flatnonzero(tg_all == "non-targeting"),
                              VCC_CTRL_CELLS, replace=False)
        ctrl = thin(read_rows(ntc_rows), VCC_UMI, rng)
        ctrl_path.unlink(missing_ok=True)
        ad.AnnData(X=sp.csr_matrix(ctrl),
                   var=pd.DataFrame(index=pd.Index(genes))).write_h5ad(
                       ctrl_path, compression="gzip")

    if not ctrl_path.exists():
        _write_ctrl()
    try:
        ref = ControlRef.load(ctrl_path, list(genes))
    except OSError:                      # 上一轮写残的缓存：重建一次再试
        print("  ⚠️ 对照缓存读不开，重建")
        _write_ctrl()
        ref = ControlRef.load(ctrl_path, list(genes))

    e2s = pd.read_csv(MAPCSV).dropna()
    gw_cols = [str(c) for c in pd.read_csv(DATA / "nadig2025" / "K562GW_p.csv.gz",
                                           index_col=0, nrows=1).columns]
    cats = sorted(set(tg_all) - {"non-targeting"})
    usable = sorted(set(gw_cols) & set(cats))
    real_n = {p: int((tg_all == p).sum()) for p in usable}
    cand = sorted([p for p in usable if real_n[p] >= VCC_PERT_CELLS], key=lambda p: real_n[p])
    pick = [cand[int(i)] for i in np.linspace(0, len(cand) - 1, N_PERT).round()]

    def rd(n):
        d = pd.read_csv(DATA / "nadig2025" / f"{n}.csv.gz", index_col=0,
                        usecols=["Unnamed: 0"] + pick,
                        dtype={c: np.float32 for c in pick}, engine="c")
        d.index = d.index.astype(str)
        return d[pick]

    lfc_s, se_s = rd("K562GW_lfc"), rd("K562GW_se")
    se_s = se_s.reindex(index=lfc_s.index)
    sym = pd.Index(lfc_s.index.map(dict(zip(e2s.ensembl, e2s.symbol))))
    keep = pd.notna(sym)
    lfc_s, se_s, sym = lfc_s[keep], se_s[keep], sym[keep]
    dd = ~sym.duplicated()
    lfc_s, se_s, sym = lfc_s[dd], se_s[dd], sym[dd]
    gate_sym = genes[np.asarray(ref.gidx)]
    common = pd.Index(gate_sym).intersection(sym)
    gi_g = pd.Index(gate_sym).get_indexer(common)
    gi_s = sym.get_indexer(common)
    B = lfc_s.to_numpy().astype(np.float64)[gi_s]
    S = se_s.to_numpy().astype(np.float64)[gi_s]
    Bc = np.where(np.isfinite(B), B, 0.0)
    Bd = Bc - Bc.mean(1, keepdims=True)
    return ref, genes, gate_sym, pick, gi_g, B, S, Bd


def design(ref, gi_g, B, S, Bd, j, variant, ramp=None):
    """一个扰动的 (r_set, lfc_t, la_vec, shift)。

    variant 语义（每一条只比它的基准多**一个**旋钮，F29 纪律）：
      v8   召集集 288，标量 shift=0.10（现状基准）
      v13  v8 + 集外一阶矩通道（原幅）
      v14  v13 + 等比压幅
      v15  **v8 + 按置信度单调的 shift 向量**（只改头部次序，不动集外通道）
      v16  v13 + 同一个 shift 向量（两个旋钮一起，仅在 v15 成立后才看）

    `ramp = (hi, lo)`：shift 按置信度 z = |b|/SE 降序从 hi 线性降到 lo。
    取 hi + lo = 0.20 使**均值恒等于 0.10** —— 于是 v15 相对 v8 只改次序，
    不改「显著性总量」，避免把两个效应混进一个观测里（T11）。
    """
    b, s = B[:, j], S[:, j]
    good = np.isfinite(b) & np.isfinite(s) & (s > 0)
    score = np.where(good, np.abs(b), -np.inf)
    sel = np.argsort(score)[::-1][:min(K_CALL, int(good.sum()))]
    r_set, lfc_t = gi_g[sel], LAMBDA * b[sel]

    shift = 0.10
    if variant in ("v15", "v16"):
        hi, lo = ramp if ramp is not None else (0.14, 0.06)
        z = np.abs(b[sel]) / s[sel]               # 源侧 Wald z = 我们的置信度
        rank = np.empty(sel.size, dtype=float)
        rank[np.argsort(z)[::-1]] = np.arange(sel.size)   # 0 = 最有信心
        shift = hi - (hi - lo) * (rank / max(sel.size - 1, 1))

    if variant in ("v8", "v15"):
        return r_set, lfc_t, None, shift
    off = Bd[:, j].copy()
    off[sel] = 0.0
    if K_OFF < off.size:
        off[np.argsort(np.abs(off))[::-1][K_OFF:]] = 0.0
    if variant == "v14":
        mn, mx = float(np.abs(lfc_t).min()), float(np.abs(off).max())
        if mx > 0.0:
            off *= (CAP_MARGIN * mn) / mx
        assert np.abs(off).max() < mn
    la = np.zeros(ref.G)
    la[gi_g] = LAMBDA_OFF * off
    return r_set, lfc_t, la, shift


def real_side(ref, gate_sym, pick):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["pick"]) == list(pick):
            return z["p_adj"], z["lfc"]
    with h5py.File(REAL, "r") as f:
        tg = as_str(f["obs"]["target_gene"])
        X = f["X"]
        dense = not isinstance(X, h5py.Group)
        pa, lf = [], []
        for p in pick:
            rows = np.flatnonzero(tg == p)
            if dense:
                M = X[rows.min():rows.max() + 1, :][rows - rows.min()]
            else:
                ip, ix, dt = X["indptr"][:], X["indices"], X["data"]
                M = np.zeros((rows.size, ref.n_genes), np.float32)
                for i, r in enumerate(rows):
                    a, b2 = ip[r], ip[r + 1]
                    M[i, ix[a:b2]] = dt[a:b2]
            M = np.asarray(M, np.float64)
            M *= (TS_CELL / M.sum(1))[:, None]
            _, q, l = de_full(ref, M)
            pa.append(q); lf.append(l)
            print(f"  real {p:10s} 显著 {(q < ALPHA).sum():5d}/{ref.G}")
    pa, lf = np.array(pa), np.array(lf)
    np.savez(CACHE, p_adj=pa, lfc=lf, pick=np.array(pick))
    return pa, lf


def run_variant(ref, gate_sym, pick, gi_g, B, S, Bd, p_adj_r, lfc_r,
                variant, ramp=None, verbose=True, checked=[]):
    """返回 (mean_reach, rows)。`rows` 每项 = (扰动, reach, k*, N_conf, n_sig, top1命中, head)。"""
    rows = []
    for j, p in enumerate(pick):
        r_set, lfc_t, la, shift = design(ref, gi_g, B, S, Bd, j, variant, ramp)
        C = design_cells(ref, r_set, lfc_t, n_cells=VCC_PERT_CELLS,
                         seed=SEED, shift=shift, lfc_all=la)
        pp, pap, lp = de_full(ref, np.asarray(C, np.float64))
        if not checked:
            q_ref, l_ref = ref.de_table(np.asarray(C, np.float64))
            assert np.allclose(pap, q_ref, atol=0, rtol=1e-12), "p_adj 与 de_table 不一致"
            assert np.allclose(lp, l_ref, atol=0, rtol=1e-12), "lfc 与 de_table 不一致"
            checked.append(True)
            if verbose:
                print("  ✅ de_full 对 de_table 逐位一致（并列项可加分解正确）")
        rc, ks, nc, head = reach_one(gate_sym, p_adj_r[j], lfc_r[j], pp, pap, lp, p)
        n_sig = int((pap < ALPHA).sum())
        top1 = head[0][4] if head else False
        rows.append((p, rc, ks, nc, n_sig, top1, head))
        if verbose:
            n_off = int(np.count_nonzero(la)) if la is not None else 0
            n_sig_off = int((pap[la != 0.0] < ALPHA).sum()) if la is not None else 0
            print(f"  {p:10s} k*={ks:5d} N_conf={nc:5d} reach={rc:.4f}  "
                  f"top1={'✓' if top1 else '✗'}  预测显著 {n_sig:5d}"
                  f"（集外 {n_sig_off:4d}/{n_off}）")
    return float(np.nanmean([r[1] for r in rows])), rows


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    t0 = time.time()
    ref, genes, gate_sym, pick, gi_g, B, S, Bd = load_pipeline()
    print(f"gate {ref.G:,}  对照 {ref.n_ctrl:,}  扰动 {pick}")
    print(f"共同基因 {gi_g.size:,}   ({time.time()-t0:.0f}s)")
    print("\nreal 侧 DE（冻结文件 E27/real.h5ad，官方每次打分用的同一份）")
    p_adj_r, lfc_r = real_side(ref, gate_sym, pick)
    args = (ref, gate_sym, pick, gi_g, B, S, Bd, p_adj_r, lfc_r)

    if mode == "gsign":
        # `artifact` 证明 reach 由「全局常数符号 × 该扰动的方向纯度」决定。那么把全局
        # 符号从副作用改成**有意选择**能拿到多少？上界由 raw reach 的 purity 门 0.9 决定：
        # 猜对方向时纯度 = q（`_reference_stats` 的多数符号率），q < 0.9 的扰动即使方向
        # 猜对也只能靠短前缀的运气。本模式同时量（a）q 的分布、（b）源侧能否预测方向、
        # （c）方向全猜对时的 oracle reach —— 三个都是纯算术，不生成细胞。
        print("\n" + "=" * 96)
        print("全局符号这个杠杆的上界 —— q 分布 / 源侧预测力 / oracle reach")
        print("=" * 96)
        print(f"{'扰动':10s} {'N_conf':>7s} {'q_real':>8s} {'源侧多数':>9s}"
              f" {'real 多数':>9s} {'猜对?':>6s} {'oracle k*':>10s} {'oracle reach':>13s}")
        print("-" * 96)
        tot, hit = [], 0
        for j, p in enumerate(pick):
            b, s = B[:, j], S[:, j]
            qr, lr = p_adj_r[j], lfc_r[j]
            pool = (qr < ALPHA) & np.isfinite(lr) & (lr != 0.0) & (gate_sym != p)
            n_conf = int(((qr < ALPHA) & (gate_sym != p)).sum())
            lp_ = lr[pool]
            f_neg = float((lp_ < 0).mean())
            q_real = max(f_neg, 1.0 - f_neg)
            real_dir = -1 if f_neg > 0.5 else 1
            # 源侧预测：可用基因上 b 的多数符号（不看 real，纯源侧量）
            ok = np.isfinite(b) & np.isfinite(s) & (s > 0) & (b != 0.0)
            src_dir = -1 if float((b[ok] < 0).mean()) > 0.5 else 1
            got = src_dir == real_dir
            hit += int(got)
            # oracle：方向猜对 ⇒ 纯度恒为 q_real（同号位移把整池同方向）。
            # k* = 池子长度 if q_real >= 0.9 else 短前缀（用逐位累积纯度算，不外推）
            sgn = np.sign(lp_) == real_dir
            d = np.arange(1, sgn.size + 1)
            pur = np.cumsum(sgn) / d
            okk = pur >= PURITY_FLOOR
            k_or = int(d[okk].max()) if okk.any() else 0
            r_or = k_or / n_conf if n_conf else np.nan
            tot.append(r_or)
            print(f"{p:10s} {n_conf:7d} {q_real:8.4f} {src_dir:+9d} {real_dir:+9d}"
                  f" {'✓' if got else '✗':>6s} {k_or:10d} {r_or:13.4f}")
        print("-" * 96)
        print(f"源侧方向预测命中 {hit}/8   oracle reach（方向全对、且真实池按 real 自身"
              f"顺序）= {float(np.nanmean(tot)):.4f}")
        print(f"对比：V8 实测 {OFFICIAL['v8']:.4f}（其中 MAT2A 一项贡献 0.1250）"
              f"   V13 {OFFICIAL['v13']:.4f}")
        print("\n判读：oracle 是**乐观上界**（它把池子按 real 自己的顺序排，我们排不出这个"
              "\n      顺序）。若上界本身就只比 V8 高一点，则「有意选全局符号」只能把伪影"
              "\n      变成可迁移的同等收益，不能放大 —— 那就该把预算挪去别的成员。")
        print(f"\n总耗时 {time.time()-t0:.0f}s")
        return

    if mode == "artifact":
        # V8 在 MAT2A 上 k*=5 / purity 5/5 / reach=1.0000，而源侧符号一致率只有 1/6。
        # 源符号几乎全错却全部命中 ⇒ 命中的符号不来自我们的生物学。唯一的其他来源是
        # 逐行重归一（decoder.py:158 `V *= TS_CELL/V.sum(1)`）给未触碰基因加的同号位移：
        # 288 个两点尖峰占掉行和，其余每个基因被同一个因子压下去。本模式直接验证。
        print("\n" + "=" * 96)
        print("V8 的 reach 是不是重归一符号巧合？—— 逐扰动拆开")
        print("=" * 96)
        for j, p in enumerate(pick):
            r_set, lfc_t, la, shift = design(ref, gi_g, B, S, Bd, j, "v8")
            C = design_cells(ref, r_set, lfc_t, n_cells=VCC_PERT_CELLS,
                             seed=SEED, shift=shift, lfc_all=la)
            _, pap, lp = de_full(ref, np.asarray(C, np.float64))
            in_call = np.zeros(ref.G, bool)
            in_call[r_set] = True
            # 未触碰基因的预测 lfc 符号分布：若几乎全同号，就是重归一的指纹
            unt = ~in_call & np.isfinite(lp) & (lp != 0.0)
            n_neg = int((lp[unt] < 0).sum())
            frac = n_neg / max(int(unt.sum()), 1)
            # real 显著且可裁决的池子里，有多少落在召集集内
            qr, lr = p_adj_r[j], lfc_r[j]
            pool = (qr < ALPHA) & np.isfinite(lr) & (lr != 0.0) & (gate_sym != p)
            n_pool_call = int((pool & in_call).sum())
            # 池子里 real 为负的比例：与未触碰基因的同号方向比对
            real_neg = float((lr[pool] < 0).mean()) if pool.sum() else np.nan
            print(f"  {p:10s} 未触碰基因 {int(unt.sum()):5d} 中 lfc<0 占 {frac:6.4f}"
                  f" | real 池 {int(pool.sum()):5d}（召集集内仅 {n_pool_call:3d}）"
                  f" real<0 占 {real_neg:6.4f}")
        print("-" * 96)
        print("判读：若「未触碰基因 lfc<0 的比例」≈1（同号指纹），且某扰动的 real 池"
              "\n      几乎全为负、且池内召集集成员极少，则该扰动的纯前缀完全由重归一"
              "\n      的同号位移产生 —— 是符号巧合，不是技巧，不可迁移到别的 panel。")
        print(f"\n总耗时 {time.time()-t0:.0f}s")
        return

    if mode == "zsign":
        # 上游前提：置信度斜坡只有在「源侧 Wald z 能预测符号可迁移性」时才可能有用。
        # F33 只量了汇总符号一致率 51–57%，没有按 z 分层 —— 没分层就无法判断
        # 头部（高 z）是否比整体更准，而 reach 的 k*>=1 完全押在头部那一条上。
        print("\n" + "=" * 96)
        print("源侧 Wald z = |beta|/SE 对符号可迁移性的分层实测")
        print("=" * 96)
        Zs, agree, wts = [], [], []
        for j, p in enumerate(pick):
            b, s = B[:, j], S[:, j]
            ok = np.isfinite(b) & np.isfinite(s) & (s > 0) & (b != 0.0)
            lr = lfc_r[j][gi_g]                 # real 侧 lfc，对齐到 common 基因
            qr = p_adj_r[j][gi_g]
            adj = ok & np.isfinite(lr) & (lr != 0.0) & (qr < ALPHA)   # 可裁决且 real 显著
            Zs.append(np.abs(b[adj]) / s[adj])
            agree.append((np.sign(b[adj]) == np.sign(lr[adj])).astype(float))
            wts.append(np.full(int(adj.sum()), 1.0))
            print(f"  {p:10s} 可裁决且 real 显著 {int(adj.sum()):5d}"
                  f"  符号一致率 {agree[-1].mean():.4f}")
        Z = np.concatenate(Zs); A = np.concatenate(agree)
        print("-" * 96)
        print(f"汇总 {Z.size:,} 对，符号一致率 {A.mean():.4f}"
              f"（F33 报的 51–57% 区间）")
        print(f"\n{'z 分位':>12s} {'n':>7s} {'z 下界':>9s} {'符号一致率':>10s}")
        print("-" * 44)
        qs = [0, 50, 75, 90, 95, 99, 99.5, 100]
        for a, b2 in zip(qs[:-1], qs[1:]):
            lo_, hi_ = np.percentile(Z, a), np.percentile(Z, b2)
            m = (Z >= lo_) & (Z <= hi_)
            if m.sum():
                print(f"{f'{a}-{b2}':>12s} {int(m.sum()):7d} {lo_:9.3f} {A[m].mean():10.4f}")
        print("-" * 44)
        print("判读：若最高 z 分位的一致率并不高于汇总值，则置信度→p_adj 的斜坡"
              "\n      无法把正确的那一条送到头部，reach 这条路结构上走不通 —— "
              "\n      该结论必须写进 RESULT，且要停止在 reach 上继续花预算。")
        print(f"\n总耗时 {time.time()-t0:.0f}s")
        return

    if mode == "sweep":
        # T10：不许用两个点判一条曲线。hi+lo 恒为 0.20 ⇒ 均值恒为 0.10，只有次序在动。
        print("\n" + "=" * 96)
        print("shift 斜坡扫描（hi+lo=0.20 ⇒ 均值恒 0.10；hi=lo=0.10 即 V8 基准）")
        print("=" * 96)
        print(f"{'ramp (hi,lo)':>16s} {'reach':>8s} {'Δ vs V8':>9s} {'top1 命中':>10s}"
              f" {'k*>0 的扰动':>12s}")
        print("-" * 62)
        for hi in (0.10, 0.11, 0.12, 0.14, 0.16, 0.18, 0.20):
            lo = 0.20 - hi
            v = "v8" if hi == 0.10 else "v15"
            mean, rows = run_variant(*args, v, ramp=(hi, lo), verbose=False)
            n1 = sum(1 for r in rows if r[5])
            nk = sum(1 for r in rows if r[2] > 0)
            print(f"{f'({hi:.2f},{lo:.2f})':>16s} {mean:8.4f} "
                  f"{mean-OFFICIAL['v8']:+9.4f} {f'{n1}/8':>10s} {f'{nk}/8':>12s}")
        print("-" * 62)
        print("判读：reach 的 k*>=1 充分条件是第一条命中（direction.py:946-949）。"
              "\n      若 top1 命中数随斜坡变宽单调上升，则置信度→p_adj 这条通道成立。")
        print(f"\n总耗时 {time.time()-t0:.0f}s")
        return

    variants = ["v8", "v13", "v14"] if mode == "validate" else [mode]
    for variant in variants:
        print("\n" + "=" * 96)
        print(f"变体 {variant}")
        print("=" * 96)
        mean, rows = run_variant(*args, variant)
        print("-" * 96)
        tag = ""
        if variant in OFFICIAL:
            tag = f"   官方实测 {OFFICIAL[variant]:.4f}   差 {mean-OFFICIAL[variant]:+.4f}"
        print(f"本地 reach = {mean:.4f}{tag}   top1 命中 "
              f"{sum(1 for r in rows if r[5])}/8")
        print("\n  头部 12 条（扰动 " + rows[0][0] + "）：基因 / 显著 / p_adj / |lfc| / 命中")
        for g, sg, q, al, mh in rows[0][6]:
            print(f"    {g:12s} {'sig' if sg else '   '} {q:10.3e} {al:8.4f} "
                  f"{'✓' if mh else '✗'}")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
