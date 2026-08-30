"""E07 的核心度量：跨细胞系响应基因集重叠率，以及它相对同系 replicate 的比值。

这个模块与数据源无关 —— 只要给它「(细胞系, 扰动) → 细胞级计数矩阵」，
它就按官方打分器的同一套判定算出 h_cross / h_replicate。

设计要点（每一条都有理由，去掉任何一条测出的数就不可用）：

1. **降采样到本届条件**：每扰动 400 个细胞、中位约 20,000 UMI。
   Replogle / Jiang 的功效远高于此；不控制会系统性高估重叠率。
2. **同一套判定**：5 CPM 门 + Wilcoxon 秩和 + 每扰动内 BH，直接用 vcclab.scorer。
3. **必须同时测同系 replicate**：每个细胞系内部对半劈开，作为分母。
   报的是比值 h_cross / h_replicate，不是绝对值 —— 因为绝对值随功效变化，
   而官方的 1 分锚点本身就是 replicate 水平。
4. **h 的定义与官方 jac 一致可换算**：jac = h/(2−h) 当 |R̂|=|R|=n。
   我们直接报 h（对称化的交集/均值大小），并同时报 jac 以便和官方锚点对齐。

判定门（见 docs/07）：
    h_cross / h_replicate >= 0.35  → 计划成立
    0.23 ~ 0.35                    → 勉强追平
    < 0.23                         → DE 半边没有可迁移信号
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VCC_CELLS = 400
VCC_MEDIAN_UMI = 20_000
H_REPLICATE_OFFICIAL = 0.570   # 由官方 r_jac=0.399 反推
H_TIE_LEADER = 0.134


def downsample_to_vcc(counts: np.ndarray, rng: np.random.Generator,
                      n_cells: int = VCC_CELLS,
                      median_umi: int = VCC_MEDIAN_UMI) -> np.ndarray:
    """把一个 (细胞 × 基因) 计数块降到本届条件。

    两步：先抽 n_cells 个细胞（不足则有放回），再对每个细胞做二项稀释，
    使中位文库大小落到 median_umi。二项稀释是计数数据正确的降深度方式
    （多项抽样的边缘分布），不是按比例缩放后取整。
    """
    n = counts.shape[0]
    idx = rng.choice(n, n_cells, replace=n < n_cells)
    sub = counts[idx].astype(np.int64)

    lib = sub.sum(1)
    cur = np.median(lib)
    if cur <= median_umi:
        return sub.astype(np.float32)          # 已经比目标浅，不上采样
    p = median_umi / cur
    out = rng.binomial(sub, min(p, 1.0)).astype(np.float32)
    return out


def responder_set(ref, counts: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """用官方同一套判定求显著响应基因集（返回 gate 内下标的布尔数组）。

    ref 是 vcclab.scorer.ControlRef（该细胞系自己的对照）。
    counts 必须是整数、且每行和为 1e6（由 vcclab.decoder.hamilton 保证），
    否则打分器的归一化不是恒等映射。
    """
    padj, _ = ref.de_table(counts, tie_correct=True)
    return padj < alpha


def overlap_h(set_a: np.ndarray, set_b: np.ndarray) -> float:
    """对称化的重叠率 h = |A ∩ B| / mean(|A|, |B|)。

    为什么用均值而不是 |A|：两个集合大小可能差很多（不同细胞系功效不同），
    用其中一个当分母会引入方向性偏差。这个定义在 |A|=|B| 时与
    官方 jac = h/(2−h) 的 h 完全一致。
    """
    na, nb = int(set_a.sum()), int(set_b.sum())
    if na == 0 or nb == 0:
        return float("nan")
    return float((set_a & set_b).sum()) / (0.5 * (na + nb))


def h_to_jac(h: float) -> float:
    """|R̂|=|R| 时 jac 与 h 的换算，用来和官方锚点对齐。"""
    return h / (2 - h)


@dataclass
class PertResult:
    pert: str
    h_cross: float
    h_rep_a: float
    h_rep_b: float
    n_a: int
    n_b: int

    @property
    def ratio(self) -> float:
        """h_cross 相对同系 replicate 的比值 —— E07 真正要报的数。"""
        denom = np.nanmean([self.h_rep_a, self.h_rep_b])
        return float("nan") if not denom else self.h_cross / denom


def measure_one(pert: str, counts_a: np.ndarray, counts_b: np.ndarray,
                ref_a, ref_b, rng: np.random.Generator,
                n_splits: int = 5) -> PertResult:
    """一个扰动、两个细胞系：算 h_cross 与两侧的 h_replicate。

    h_replicate 用「对半劈开、各自与对方比」的方式估，重复 n_splits 次取均值 ——
    与官方 replicate 锚点的构造方式一致（官方用 5 个不相交半分）。
    """
    from vcclab.decoder import hamilton

    def prep(c):
        d = downsample_to_vcc(c, rng)
        d *= (1e6 / np.maximum(d.sum(1, keepdims=True), 1))
        return np.vstack([hamilton(r) for r in d]).astype(np.float32)

    a, b = prep(counts_a), prep(counts_b)
    ra, rb = responder_set(ref_a, a), responder_set(ref_b, b)

    def rep(block, ref):
        vals = []
        for _ in range(n_splits):
            perm = rng.permutation(block.shape[0])
            half = block.shape[0] // 2
            s1 = responder_set(ref, block[perm[:half]])
            s2 = responder_set(ref, block[perm[half:2 * half]])
            vals.append(overlap_h(s1, s2))
        return float(np.nanmean(vals))

    return PertResult(pert, overlap_h(ra, rb), rep(a, ref_a), rep(b, ref_b),
                      int(ra.sum()), int(rb.sum()))


def verdict(ratios: np.ndarray) -> str:
    med = float(np.nanmedian(ratios))
    if med >= 0.35:
        return f"GO — 中位比值 {med:.3f} >= 0.35，计划成立"
    if med >= 0.23:
        return f"MARGINAL — 中位比值 {med:.3f} 在 0.23~0.35，勉强追平，需同时做 pds/mse"
    return f"NO-GO — 中位比值 {med:.3f} < 0.23，DE 半边没有可迁移信号"


# ---------------------------------------------------------------------------
# 汇总统计路径（E07 实际走的这条）：只有 beta / se / p 三个矩阵，没有细胞级数据。
# ---------------------------------------------------------------------------

def fission(beta: np.ndarray, se: np.ndarray, rng: np.random.Generator,
            tau: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Data fission：把一个正态估计劈成两个**独立**的伪重复。

    给定 beta_hat ~ N(beta, se^2) 且 se 已知，取 Z ~ N(0,1)：

        beta_1 = beta_hat + tau * se * Z
        beta_2 = beta_hat - se * Z / tau

    则 Cov(beta_1, beta_2) = tau*se^2*(1/tau) - ... = se^2 - se^2 = 0，
    且二者联合正态，故**独立**（不只是不相关）。

    方差：Var(beta_1) = se^2 (1 + tau^2)，Var(beta_2) = se^2 (1 + 1/tau^2)。
    两者仅在 tau=1 时相等，此时各自 SE = se*sqrt(2) —— 正好等价于样本量减半，
    与官方 1 分锚点的 split-half 完全同构。这是必须拿到 lfcSE 文件的原因。

    返回 (beta_1, beta_2, se_1, se_2)。tau != 1 时两个 SE 不同，必须分别用。
    """
    z = rng.standard_normal(beta.shape)
    return (beta + tau * se * z,
            beta - se * z / tau,
            se * np.sqrt(1.0 + tau * tau),
            se * np.sqrt(1.0 + 1.0 / (tau * tau)))


def rank_matrix(pvals: np.ndarray) -> np.ndarray:
    """每列内按 p 升序的名次（0 = 最显著）。NaN 当作 1.0 排到最后。

    名次矩阵存 int16（被测基因数 < 32767），任意 K 的 top-K 成员就是 rank < K，
    所以扫 K 不用重复排序。
    """
    p = np.where(np.isfinite(pvals), pvals, 1.0)
    order = np.argsort(p, axis=0, kind="stable")
    rank = np.empty(p.shape, np.int16)
    rows = np.arange(p.shape[0], dtype=np.int16)[:, None]
    np.put_along_axis(rank, order, np.broadcast_to(rows, p.shape), axis=0)
    return rank


def h_topk(rank_a: np.ndarray, rank_b: np.ndarray, k: int) -> float:
    """功效匹配的重叠率：两边各取最显著的 k 个，h = |交集| / k，取扰动间中位数。

    为什么不能用 BH 阈值 + mean 分母：各系功效差一个量级（中位显著数 22–318），
    mean(|A|,|B|) 会让小集合即使完全嵌套进大集合也拿不到高分 —— 度量被功效污染。
    固定 K 两边同规模，功效差异被完全消掉。
    """
    inter = ((rank_a < k) & (rank_b < k)).sum(0).astype(np.float64)
    return float(np.median(inter / k))


def chance_overlap(k: int, n_genes: int) -> float:
    """K 个基因从 n_genes 个里随机取，两组期望重叠率 = K / n_genes。"""
    return k / n_genes


def h_topk_shuffled(rank_a: np.ndarray, rank_b: np.ndarray, k: int,
                    n_rep: int = 8) -> float:
    """置换零假设：把 B 的扰动标签错位后再比，取多次错位的均值。

    **这才是 h_cross 的正确基线，K/G 不是。** 理由：K/G 假设 top-K 是从 G 个基因里
    均匀抽的，但真实数据里存在一批「常常上榜」的基因（核糖体、翻译应激、细胞周期），
    它们在每个细胞系的大多数扰动 top-K 里都出现。这种边缘结构会制造大量跨系重叠，
    而它与「扰动特异性响应是否可迁移」毫无关系 —— 用 K/G 当基线会把它全算成信号。

    错位用随机非零循环移位实现：保证无不动点（真配对一个都不残留），
    同时完整保留每个细胞系的基因边缘上榜频率和每个扰动的 top-K 规模。
    """
    n = rank_a.shape[1]
    if n < 2:
        return float("nan")
    a_top = rank_a < k
    b_top = rank_b < k
    rng = np.random.default_rng(0)
    offsets = rng.choice(np.arange(1, n), size=min(n_rep, n - 1), replace=False)
    vals = [np.median(((a_top & np.roll(b_top, int(o), axis=1)).sum(0)) / k)
            for o in offsets]
    return float(np.mean(vals))


def ratio_vs_permutation(h_cross: float, h_cross_null: float,
                         h_rep: float, h_rep_null: float) -> float:
    """扣掉置换基线后的比值：跨系拿到的**扰动特异性**信号，占同系重复水平的几成。

    分子分母各自减掉自己的置换基线，所以「常常上榜的基因」这一项两边都被消掉。
    """
    denom = h_rep - h_rep_null
    return float("nan") if denom <= 0 else (h_cross - h_cross_null) / denom


def ratio_cc(h_cross: float, h_rep: float, k: int, n_genes: int) -> float:
    """扣掉随机基线后的比值：跨系拿到的超随机信号，占同系重复水平的几成。"""
    ch = chance_overlap(k, n_genes)
    return float("nan") if h_rep <= ch else (h_cross - ch) / (h_rep - ch)
