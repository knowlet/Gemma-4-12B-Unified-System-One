# Decision3：GX10 原生 GPU 短／長輸入驗證

日期：2026-10-25（Asia/Taipei）。固定版本的 `vllm-sr/d3-edge` 已在 GX10 的 GB10 上完成兩次自造原生呼叫。Root 另以標準函式庫重算全部 8 個 head、164 個合法機率，並核對原始執行身分與所選 owner 的清理證據。本次僅接受這兩次診斷的數值與執行結果；完整瀏覽器比較、新 109 筆訓練候選及兩套 JevBench 仍待驗證。

## 固定輸入與量測

| 項目 | 本次設定／結果 |
| --- | --- |
| 基底模型 | `vllm-sr/d3-edge`，revision `cab71e6ad3461b8086d51ddb572c656babd63b0b` |
| 裝置與套件 | NVIDIA GB10、compute capability 12.1；Torch 2.13.0+cu130、CUDA 13.0、Transformers 5.17.0 |
| 原生數值設定 | BF16 backbone、FP32 readout `[255,1024]`、temperature 1；原生 batch size 8 保持不變 |
| 實際 forward | 每次為四列 partial batch；四個問題全部執行，合法選項數依序為 7／38／36／1，包含 singleton |
| 載入 | 一次真正的 tracked provider constructor，3.344540 秒 |
| 執行次序 | 短輸入後接長輸入；沒有額外暖機、HTTP、helper 或瀏覽器執行 |
| 新訓練 delta | 未載入；本次仍是原始 Edge readout |

| 呼叫 | 各問題 token 數 | 完整呼叫秒數 | Root FP64 重算最大絕對誤差 |
| --- | --- | ---: | ---: |
| 短輸入 | 156／373／359／114 | 1.845402 | 1.785016e-7 |
| 長輸入 | 3117／4196／4579／3126 | 1.370788 | 1.202264e-7 |

完整呼叫時間包含 DTO 驗證、原生準備與 forward、CUDA 同步、全部合法輸出的重播、來源檢查、特徵與證據寫入、hook 清理及 watchdog 結束檢查，兩次均通過原 20 秒限制。輸入與候選未裁剪。

Leaf 在最後 watchdog 結束前的 durable 紀錄為 16.673004 秒；原 owner 記錄的完整 child command 為 17.979917 秒，Root 的整次 owner 執行為 19.298338 秒。Leaf 保留 120 秒整體上限；owner 原流程另保留 7 秒清理時間，因此 command deadline 最多為 113 秒。這些時間分屬不同範圍，不能互換成單次模型推論延遲。

## 獨立重算與執行核對

Root 的重算未匯入 Torch、NumPy 或其他模型框架，在單 CPU core、512 MiB、CPU 30 秒／wall 35 秒限制內完成，實際為 2.469 秒。對每個 head 以當次 final-norm 特徵與原始 FP32 readout 計算全部 255 個 FP64 `fsum` dot logits，再依完整合法選項做 softmax。

164／164 個合法機率符合原預先固定容差 `abs(replay-native) <= 1e-6 + 1e-5*abs(native)`；8／8 個 ordered argmax 與 publisher entropy confidence 均吻合。兩次捕捉的 FP32 baseline bytes 完全相同，其 NPY payload 也逐 byte 符合固定 `readout.safetensors` 的 weight body。重算範圍從當次 final-norm 特徵開始，未另跑 encoder，也未建立 CPU／GPU backbone 等價或穩定延遲分布。

Root 核對 127 個目前 leaf 來源／runtime 檔案、21 個完整載入檔案（20 個 publisher loaded files 加 manifest）及 19 個 owner 來源參照。這些是分別固定的投影，不宣稱完整 transitive DSO closure。原 command 的 argv、executable、cwd、PID／start、nonce 與 cgroup 均完成連結，並保留兩次穩定的實際程序觀察。

原 owner 成功，所選 unit 與觀測 cohort 已退出、cgroup 不存在；目前 unit／cgroup 也確認不存在。原始 `model_qualified`、`continuation_authorized`、`whole_tree_exit_confirmed`、`no_escape_enforced` 仍為 false；Root 另立本次自造原生診斷的窄範圍接受紀錄。

## 資源與失敗保留

新 owner 使用 8 GiB、swap 0、CPU quota 500%、affinity 15–19、unit runtime 120 秒與 stop 3 秒，並在各 host gate 要求 ComfyUI running／pending 皆為零。既有 CPU5／CPU10 來源與收據保持原樣。

同一 leaf 程序的 Torch allocator peak 為 allocated 3,428,315,136 bytes、reserved 3,984,588,800 bytes。這不是整機 UMA peak，也不是分別量測的每次呼叫 peak。模型、hook 與保留參照卸載後，清理目前程序 cuBLAS workspaces，GPU tensor allocation 由初始零回到零。

先前不載模型的 owner control 已驗證資源與清理，但不算模型結果。第一輪完整模型嘗試在建構後，被瀏覽器專用格式拒絕，沒有觀測到 encoder forward；該次失敗與原始清理錯誤保持封存，外層所選 unit／cohort 已清理。修正版改用正常驗證的 `Decision3NativeRequest` 直接呼叫真正 provider，保留原始四個診斷問題；正式瀏覽器 adapter 的動作格式檢查沒有更動。

公開介面回歸測試見 commit `9f4db55`：整份測試檔 47 項通過，涵蓋診斷請求在 38／255 個 target 選項下的完整序列化，以及瀏覽器入口在 backend 呼叫前拒絕不符格式的輸入。這些介面測試與本次實際 GPU 數值驗證分開記錄。

## 仍待完成的評測

| 新候選評測 | 分數 |
| --- | --- |
| 十場景 V7 瀏覽器比較 | null |
| Benchmark Heaven public231 | null |
| JevBench coherenceMINI | null |

本次沒有瀏覽器成功率、模型品質、競品勝出或 SLO 結論。下一步是獨立驗證新 109 筆 operation delta 的 GPU 套用，再執行完整十場景比較與兩套 fresh JevBench，分開報分；原 Edge 歷史分數不繼承給新候選。

公開聚合與證據雜湊見[收據](decision3-gx10-native-gpu-qualification-results.json)，方法與來源比較見[研究紀錄](../../research/jev-decision3-gx10-gpu-native-qualification.md)。權重、特徵、原始請求、機率與逐筆預測留在本機私人產物。
