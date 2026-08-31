# E10 — 用主办方自己的 2025 H1 数据实测 $\lvert R_p\rvert$

**问题**：$\lvert R_p\rvert$ 决定召集集合大小 $K_p$，是整个框架最大的杠杆，
但四个来源给出的值跨了 26 倍（7 / 11 / 36 / 288）。

**为什么只有 2025 H1 能回答**：同一批主办方 · 同一套 pilot-screen 分层选基因流程 ·
同一套实验协议（dual-guide CRISPRi + 10x Flex）· 基因面板几乎相同 ·
深度是本届 7.4 倍**因而可以降采样到本届功效**。

**方法**：降采样到本届确切条件（18,400 对照 · 400 细胞/扰动 · 20,000 UMI/细胞），
用 `vcclab.scorer.ControlRef.de_table`（E04 已验证 = 官方 `cell-eval2` preset `vcc2026`）。

**结论**：中位 **253** → F8 的 288 成立；但 IQR [7, 2418]，跨 350 倍。
26 倍的先前差异来自 **DE 方法**（Wilcoxon-on-cells 的伪重复 vs DESeq2/pseudobulk），
不是生物学也不是深度。详见 [RESULT.md](RESULT.md)。

```bash
bash download.sh                                          # 6.93 GB
~/vcc2026/.venv/bin/python experiments/E10-h1-erp/run.py  # 2149 s
```
