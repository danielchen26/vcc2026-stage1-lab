# E11 — 直接实测 $h_{\text{replicate}}$

**问题**：追平榜首所需的 $h = 0.134$ 是三层反推得来的
（官方 $r_{\text{jac}}=0.399 \to h_{\text{replicate}}=0.570 \to 0.23\times$），
链条上没有一个数是直接测的，而所有 go/no-go 判断都挂在它身上。

**方法**：字面复现官方 1 分锚点「真实数据对半劈开」——
对照 38,176 → 两个互不相交的 18,400；每扰动细胞 → 两个互不相交的 400；
UMI 稀释到 20,000；两半各自用自己的对照算 $R_p$。

**结论**：实测 $h_{\text{replicate}} = 0.550$（反推 0.570，差 4%），
$r_{\text{jac}} = 0.379$（官方 0.399，差 5%），恒等式精确成立。**锚点证实。**
详见 [RESULT.md](RESULT.md)。

```bash
bash ../E10-h1-erp/download.sh   # 共用同一份 6.93 GB 数据
~/vcc2026/.venv/bin/python experiments/E11-replicate-anchor/run.py   # 2147 s
```
