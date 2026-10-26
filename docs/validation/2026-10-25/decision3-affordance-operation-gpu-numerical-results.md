# Decision3：109 筆 operation delta 的 GX10 GPU 數值驗證

日期：2026-10-25（Asia/Taipei）。固定版本的 `vllm-sr/d3-edge` 已在 GX10 完成一次真正模型載入與 16 次原生 forward，實際載入 109 筆訓練產出的非零 operation delta。Root 以標準函式庫 FP64 重算 baseline 與 composed 兩個分支，並獨立核對來源、權重位元組與原 owner 清理。本次接受範圍限於這批呼叫的數值及執行證據；瀏覽器品質與兩套 JevBench 仍待評測。

## 輸入與獨立重算

模型 revision 為 `cab71e6ad3461b8086d51ddb572c656babd63b0b`；保留原生 batch size 8、BF16 backbone、FP32 `[255,1024]` readout、temperature 1。delta 與原 readout 分開保留，CPU→CUDA 的原始 FP32 payload bytes 完全相同，僅依實際 head ID 對 `operation` 的原始 logits 加上 `h @ delta.T`。

| 呼叫次序 | 完整輸入與角色 |
| --- | --- |
| 1 | 第一筆 train wire 的真正原生呼叫，加上 exported zero-composition 數值對照；此零值路徑未使用訓練候選 wrapper，也未冒用候選 capability |
| 2–14 | 全部 13 筆已接受的原始 train wires；operation K4／K7／K8 分別為 4／5／4 筆，head 數 M2／M4 分別為 4／9 筆 |
| 15 | 自造 M1、target K255、沒有 operation；完整原始回應保持相同 |
| 16 | 自造 M3，依序為 click target K1、operation K7、text target K255；operation 位於實際第 2 列，singleton 也真正執行 forward |

16 次呼叫的 M1／M2／M3／M4 數量為 1／5／1／9。所有原始 wire、head、criteria 與內容均保留。正常 DTO 會調整 question 外層 schema 欄位的序列化順序，因此原 transport wire hash 與 provider payload hash 分別固定；head、criteria 與巢狀內容順序沒有改動，詳見[研究紀錄](../../research/jev-operation-only-readout.md)。

兩分支各重算 50 個 head、662 個合法機率，合計 1,324 次機率比較與 100 個 ordered argmax；每個 head 均計算完整 255 個 logits。全部符合預先固定容差 `abs(replay-native) <= 1e-6 + 1e-5*abs(native)`，兩分支最大絕對誤差皆為 `2.507268498375126e-7`。所有 publisher entropy confidence 與 argmax 均吻合。

35 個非 operation 原始 answer 完全相同，缺少 operation 的自造呼叫也保留整份 raw response。各次 W0 bytes 符合固定 publisher readout；所有候選呼叫的 delta bytes 均符合訓練產物。13／13 筆 train operation 分布改變，8／13 筆 argmax 改變；本次沒有讀取標籤，這些變化不是準確度或改善證據。

## 時間、資源與執行範圍

| 量測範圍 | 秒數 |
| --- | ---: |
| 真正 provider constructor | 3.406228 |
| 首次原生呼叫與零值數值對照 | 2.011702 |
| 後續 13 筆完整 train candidate 呼叫，最小／最大 | 0.663763／0.840100 |
| 兩次自造 candidate 呼叫 | 0.858921／1.376326 |
| Leaf 最後 durable 紀錄，尚未完成 whole watchdog 清理 | 29.881475 |
| 原 owner 的完整 child command | 31.614042 |
| 整次 owner 執行 | 32.664510 |

完整單次呼叫包含 DTO、來源檢查、原生 forward、數值重播、私有 witness 寫入、CUDA fence、hook 與 watchdog 清理，均通過原 20 秒限制。13 筆候選時間接在首次零值對照之後，不是冷啟動、成對品質比較或 SLO 證據。真實 wires 的各次最長輸入為 544–921 tokens；自造呼叫寬度為 3799／3800 tokens，未驗證完整 16k context。

owner 維持 8 GiB、swap 0、120 秒整體上限，child command 上限為 113 秒。整個模型載入與 16 次呼叫的 Torch allocator peak 為 allocated **2,604,370,944 bytes**、reserved **3,714,056,192 bytes**；這不是整機 UMA、resident memory 或個別呼叫 peak。完成模型、delta 與目前程序 cuBLAS workspace 清理後，GPU tensor allocation 由零回到零。

Root 核對 28 個 parent pins、129 個 leaf pins，以及 21 個完整 publisher 載入檔案；保留兩次穩定且相符的 child 身分觀察。原所選 unit／觀測 cohort 已退出，目前 unit／cgroup 不存在。此證據不建立 no-escape 或 whole-tree exit 宣稱；原始資格旗標仍為 false。

第一次獨立 auditor 在讀取陣列前，因 fit delta reference 另含 dtype／format／shape metadata 而拒絕。第二版完整驗證該 metadata 與相同檔案身分後完成重算，原失敗與原始執行均保留。公開的[held composition 原始碼](../../../scripts/jev_decision3_operation_gpu.py)及 [18 個 stdlib 自造控制測試](../../../tests/test_jev_decision3_operation_gpu.py)另行保留；其語意核對與實際 GPU 證據分開記錄。

## 尚未完成

| 新候選評測 | 分數 |
| --- | --- |
| 完整十場景 V7 瀏覽器比較 | null |
| Benchmark Heaven public231 | null |
| JevBench coherenceMINI | null |

`model_qualified`、`browser_qualified`、`benchmark_qualified`、`continuation_authorized` 皆為 false；本次沒有瀏覽器 adapter、HTTP、標籤、準確度或競品勝出結論。這項數值接受不授予瀏覽器、benchmark 或新訓練權限，也不繼承舊模型分數。

公開聚合與 SHA256 見[機器可讀紀錄](decision3-affordance-operation-gpu-numerical-results.json)。前置證據分別見[原生短／長 GPU 驗證](decision3-gx10-native-gpu-qualification-results.md)與 [109 筆合併訓練](decision3-affordance-combined-fit-results.md)。原始請求、逐筆機率、特徵與權重留在本機私人產物。
