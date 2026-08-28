/* Numerical Recipes erfcc —— 相对误差 < 1.2e-7，足以显示到 1e-30 量级 */
const COF = [
  -1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2,
  -9.561514786808631e-3, -9.46595344482036e-4, 3.66839497852761e-4,
  4.2523324806907e-5, -2.0278578112534e-5, -1.624290004647e-6,
  1.30365583558e-6, 1.5626441722e-8, -8.5238095915e-8, 6.529054439e-9,
  5.059343495e-9, -9.91364156e-10, -2.27365122e-10, 9.6467911e-11,
  2.394038e-12, -6.886027e-12, 8.94487e-13, 1.313945e-12, -3.60148e-13,
];

export function erfc(x: number) {
  const z = Math.abs(x);
  const t = 2 / (2 + z);
  const ty = 4 * t - 2;
  let d = 0;
  let dd = 0;
  for (let j = COF.length - 1; j > 0; j--) {
    const tmp = d;
    d = ty * d - dd + COF[j];
    dd = tmp;
  }
  const ans = t * Math.exp(-z * z + 0.5 * (COF[0] + ty * d) - dd);
  return x >= 0 ? ans : 2 - ans;
}

/** 标准正态 CDF */
export const Phi = (z: number) => 0.5 * erfc(-z / Math.SQRT2);

/**
 * 双侧可检出概率。
 * 源形式：AdaptiveEROP src/Core/p_success.jl 的 compute_p_success_cdf，
 * 把 assay 成功阈值 tau 换成该 (基因, 细胞系) 的最小可检出效应 MDE。
 */
export function pDetect(mu: number, sigma: number, mde: number) {
  if (sigma <= 0) return Math.abs(mu) > mde ? 1 : 0;
  return Phi((mu - mde) / sigma) + Phi((-mde - mu) / sigma);
}

/** 是否落在阈值附近、需要非参处理（否则解析 Phi 误差 < 2%） */
export const needsNonparametric = (mu: number, sigma: number, mde: number, k = 1) =>
  sigma > 0 && Math.abs(Math.abs(mu) - mde) / sigma < k;

/** James–Stein 对角收缩：AdaptiveEROP src/HierarchicalBayes/mean_shift.jl 的对角退化 */
export const jamesStein = (rBar: number, sigma2Gene: number, sigma2Between: number, nC: number) =>
  (rBar * (nC * sigma2Between)) / (sigma2Gene + nC * sigma2Between);
