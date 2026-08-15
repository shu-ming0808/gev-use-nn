# RBF 與 Matérn 方向性檢定

## 檢定設計

- 六組預先指定的單尾 exact paired sign-flip tests。
- 推論單位為 10 個 geographic K-means blocks。
- Original stations 的 $H_1$ 為 Matérn 較好；gridded outcomes 的 $H_1$ 為 RBF 較好。

## 結果

| Outcome | RBF RMSE | Matérn RMSE | $H_1$ | Raw $p$ | Holm $p$ | 結論 |
|---|---:|---:|---|---:|---:|---|
| Original annual parameters | 1.7583 | 1.7521 | Matérn | 0.3975 | 1.0000 | 未拒絕 $H_0$ |
| Original monthly parameters | 0.7931 | 0.5851 | Matérn | 0.0010 | 0.0059 | 拒絕 $H_0$ |
| Gridded annual parameters | 1.6962 | 1.6652 | RBF | 0.8398 | 1.0000 | 未拒絕 $H_0$ |
| Gridded monthly parameters | 0.9883 | 0.9128 | RBF | 0.9902 | 1.0000 | 未拒絕 $H_0$ |
| Gridded monthly $RL_{50}$ | 2.1882 | 1.9535 | RBF | 0.9355 | 1.0000 | 未拒絕 $H_0$ |
| Gridded monthly $RL_{100}$ | 2.5910 | 2.2617 | RBF | 0.9609 | 1.0000 | 未拒絕 $H_0$ |
