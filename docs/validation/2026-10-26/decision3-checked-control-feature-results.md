# Decision3 GX10：checked-control 原生特徵收集

日期：2026-10-26（Asia/Taipei）。兩個分批執行的原生 CPU 工作均以 exit code 0 結束，**10 筆 train-only 完整請求、40 個問題 head、141 個合法機率位置**已通過獨立核對。這是供後續訓練使用的特徵證據；119 筆 operation 候選也已完成固定 512 步訓練與產物完整性核對；模型部署數值、瀏覽器與兩套 JevBench 的新成績仍未驗證。

| 分批 | 完整請求 | 問題 head | 合法機率位置 | 原始啟動至收尾全程 |
| --- | ---: | ---: | ---: | ---: |
| 第一批 | 8 | 32 | 113 | 74.02052065404132 秒 |
| 第二批 | 2 | 8 | 28 | 30.014974675956182 秒 |
| 合計 | 10 | 40 | 141 | 兩批各自啟動與收尾 |

每筆請求均保留原始 K6 operation 選項、全部四個 head、原始選項順序與完整 wire；沒有移除 checked radio、補選項、重新編號或以 teacher 標籤改寫原請求。後續僅有 **10 個 operation row**可作為這批資料的監督對象，其餘 **30 個 target head**保留為執行與數值證據。標籤與 feature-only 輸入分開，原始瀏覽器 authoring／含標籤 projection 不作為本次原生特徵擷取輸入。

兩批沿用 GX10 的五核心 CPU、4 GiB 記憶體與零 swap 範圍，以及每次 20 秒、每批 120 秒限制。原始 BF16 encoder 與 FP32 readout 保持原值。完整 native call 的最小／中位數／最大時間為 **6.974492524983361／7.4828501054726075／7.934742737037595 秒**；這包含該次原生準備、推論、數值核對與必要留證。表列全程時間另含各批啟動和收尾。這些時間未建立瀏覽器端到端期限或穩定 SLO。

獨立稽核逐一連結實際請求、完整 head 與選項順序、同一次 encoder forward 的特徵、原生回傳、W0 和來源。原生 Torch、NumPy 零 delta 重播及獨立 stdlib FP64 dot-product／softmax 重算均通過，所有原始 ordered argmax 一致。141 個合法機率位置的最大絕對誤差為 `1.3444587606548097e-6`，使用原本的混合容許誤差：絕對 `1e-6` 加相對 `1e-5`，未放寬門檻。每批另核對 22 個所選小型來源檔案；原有完整來源、模型與 runtime 身分核對保留。

先前兩次嘗試均保留並排除於接受的特徵集合：一次在建立模型前，因 owner base source 與所選來源重複而拒絕；另一次雖擷取八筆，但啟動後立即讀取的 `/proc` argv 為空，缺少完整實際執行參數證據，因此未接受。未以重建 argv 或事後改寫原紀錄補足該次證據。

本次額外要求執行期間的唯讀 OS 觀測：兩次 live process 讀值須維持相同 PID、啟動時間、process group、session、UID 與 cgroup，並連結完整十個 argv、resolved executable、cwd 及原始 NUL 分隔參數。觀測時間窗必須位於該原始命令的開始與結束之間。早期 argv 僅允許原值為空或與完整所選命令一致；兩種情況都必須取得新的完整 live 觀測，非空錯誤值直接拒絕。

兩個原始 owner 的範圍收尾均通過，當前所選 unit 已不存在。這項證據只涵蓋所選 unit 與觀測到的程序群；原始 `model_qualified`、`continuation_authorized`、`whole_tree_exit_confirmed` 與 `no_escape_enforced` 旗標保持 false，未宣稱 OS containment。特徵接受紀錄的 `fit_authorized` 仍為 false，後續 optimizer 必須有獨立選定的訓練能力與來源。

| 固定證據 | SHA256 |
| --- | --- |
| 原生 feature collector | `182212b242cf4e00e9841055c3ea16e5b671af1042f8ff59a67d8994345da2a5` |
| 獨立 feature auditor | `3e300a3f73b9f375b07850e9c31c6fb9ff6073711ed2ce5df8cec5456dd8962b` |
| W0 原始 FP32 bytes | `d408d94a0ab2915fa85c511dd9b8d62af49903c4e4ca28e2e2d5aff9ef565e0f` |
| 特徵集合接受紀錄 | `3ac7d214402c7828edeb0f76d2ab8d1f1df7ebd4b1137f5943895021259d72c3` |

119 筆 operation 候選由原有 109 筆與新 10 筆組成，K4／K6／K7／K8 分別為 4／10／5／100 筆。原有 96＋13 筆證據鏈重新核對後，從零 delta 執行固定 512 步，未使用前次 delta 暖啟動。訓練至最後寫入前耗時 **2.2827 秒**，原始工作啟動至收尾 **4.0075 秒**。訓練 loss 從 **1.5916 降至 0.9743**；這是訓練資料上的觀測，尚無驗證集或瀏覽器成功率證據。

獨立 stdlib 核對重建了完整 109＋10 筆 row mapping，重新驗證 47 個來源與原始程序／範圍收尾，並以既有 NPY 讀取器檢查有限的 FP32 `255×1024` delta。第 8 至 254 列維持精確零值，delta L2 norm 為 `0.8079483741699617`，結尾重新核對了產物雜湊。這證明產物完整性；尚未獨立重跑 optimizer、驗證部署時的合成機率，或測得新的模型品質分數。

原先 [109-row 候選的首案例配對](../2026-10-25/decision3-firstcase-gpu-paired-results.md)仍為候選與 NF4 對照各 **0/1**，不因本次特徵收集而更新。這批範例的動機與排除界線見 [checked-control 訓練設計](../../research/jev-checked-control-progress.md)。119-row 候選的完整瀏覽器、Benchmark Heaven 與 JevBench coherence 分數均為 null。原始 Edge 的 [Heaven 68.83%](../2026-10-25/decision3-native-heaven-gpu-results.md)及 [coherence 69.64052287581698／100](../2026-10-25/decision3-native-coherence-gpu-results.md)是另外兩份評測，不能轉用為新候選的成績。

聚合欄位見[機器可讀紀錄](decision3-checked-control-feature-results.json)。公開紀錄不含原始目標、wire、機率陣列、標籤、程序識別資訊或私人產物位置。
