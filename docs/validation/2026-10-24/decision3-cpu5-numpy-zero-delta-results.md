# Decision3 CPU5：NumPy 零差異驗證

本次完成真正模型載入、四組決策的同一次前向特徵擷取，以及 Torch／NumPy 零差異重播。這是訓練前的數值與執行環境驗證，尚未訓練，也沒有品質提升或勝出宣稱。

前一次 actual04 在原生決策呼叫前失敗：過寬的 `blas` 名稱規則將 GPU 函式庫納入 CPU 雜湊容量限制。新目錄保留原失敗來源，只將明確 CUDA BLAS 檔名移至獨立的 stat-only 清單；NVPL 仍留在 CPU 清單。cuBLAS 是 [NVIDIA 的 GPU BLAS 函式庫](https://docs.nvidia.com/cuda/cublas/index.html)，NVPL 是 [aarch64 CPU 數學函式庫](https://docs.nvidia.com/nvpl/latest/)。映射紀錄不代表已證明執行或符號綁定；沒有宣稱完整 DSO 相依性閉合。

256 MiB 單檔、512 MiB CPU 清單、CPU＋GPU 合計 24 檔、單次 20 秒、整體 120 秒、4 GiB 記憶體與既定數值容差全部保留。

| 項目 | actual05 結果 |
|---|---:|
| 原始 root／owner | exit 0／completed；本次 scoped closure 通過 |
| 模型載入／真正決策呼叫 | 1／1 |
| 同次 first-layer／final-norm | 均為 `[4,188,1024]`，各一次 |
| 合法選項數 | 8、1、2、3；順序與 argmax 全部一致 |
| NumPy 版本 | 2.5.2，Python 3.13.14 |
| 完整單次決策 | 3.220928 秒 |
| 整體 actor／root | 15.150610／17.160415 秒 |
| owned unit 記憶體峰值 | 1,094,119,424 bytes；無 memory events |
| CPU 雜湊清單 | 22 檔，共 94,612,175 bytes |
| GPU stat-only 清單 | 2 檔；未讀取／雜湊其內容 |

四組 Torch 及 NumPy 合法機率都符合原先的 `abs(replay-native) <= 1e-6 + 1e-5*abs(native)`。NumPy 最大絕對誤差為 1.00248548e-6，仍符合這個混合容差；沒有改成較寬門檻。單選項信心為 1。Root 另以純標準函式庫讀取有限 FP32 NPY，核對原始位元組雜湊、形狀與所有合法槽位的獨立 float64 累加重播。

Root 核對了 58 筆選定來源、20 個模型檔、28 個 runtime 身分（22 個雜湊、6 個 stat-only）、原始程序身分與本次結束紀錄。其他審閱者僅讀來源，沒有第二位實際資料審閱者。28 個來源回歸測試通過。診斷特徵與 W0 永久維持 `PENDING_NONTRAINING`，不送入 optimizer；後續訓練資料須另行收集並經 root COMMITTED 驗收。

兩套 JevBench 分開保留：準確度評測 Decision3 為 **158/231（68.40%）**，Gemma 為 **195/231（84.42%）**；機率一致性 Decision3 先前只完成 **1234/1248**，因原 10,000 秒逾時而**不報分**，Gemma 為 **41.26/100**。本次沒有變更任何評測分數。先前真正 HTTP＋Chromium 診斷仍因完成狀態判斷失敗；接下來用未接觸評測答案的 train-only 特徵訓練，再驗證實際改善。

完整可公開彙總見 [JSON](decision3-cpu5-numpy-zero-delta-results.json)。原始模型、特徵、catalog 內容、機率回答及 actor state 留在本機忽略目錄。
