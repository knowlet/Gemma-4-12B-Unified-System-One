# Decision3：雙頭候選通用路徑的 coherence MINI 評測

日期：2026-10-27（Asia/Taipei）。在 GX10 以全新執行結果與本次專用的記憶體快取，完成官方 JevBench coherence MINI 的全部公開人口。官方 scorer 的機率一致性為 **69.6405／100**，95% case bootstrap 區間為 **67.0155–71.9375／100**。原 worker 及監督器均回報完成，獨立 CPU 稽核重跑原官方 scorer 並核對一致；原始結果仍標示為待外部接受。

這次 1,248 個不同請求均使用 Decision3 Edge 原生通用路徑。瀏覽器候選的 `operation` 與 `click_target` 兩個增量頭全程明確 bypass，所以此分數描述原生通用模型的一致性，不能當成增量頭效果或瀏覽器任務完成率。

| 官方 coherence MINI 指標 | 本次結果 |
| --- | ---: |
| Overall，五個維度等權平均 | 69.6405／100 |
| 95% case bootstrap 區間 | 67.0155–71.9375／100 |
| BAT：批次 | 100／100 |
| CHO：選擇 | 69.4444／100 |
| LOG：邏輯 | 68.3333／100 |
| MEA：機率測度 | 34.3137／100 |
| REP：表示 | 76.1111／100 |
| Graded overall | 91.6060／100 |
| 不完整關係／略過 outcome | 0／0 |

官方人口為 240 個案例、50 項關係與 1,200 個計畫試驗。這些試驗依官方規則產生 1,468 筆 outcome；部分試驗有多個目標，不能將 outcome 筆數當作試驗數。全部 50 項關係的 coverage 均為 100%，三類排除項均為空。執行器實際完成 **1,248 次不同的模型 callback**，另有 193 次 SDK 內部重複請求命中新建的記憶體快取；未使用先前模型答案或重試。

固定 [JevBench 官方 revision `e187336`](https://github.com/JevBench/jevbench/tree/e18733694623aa93058e279c3534f8b8e2edefa3) 及 MINI 計畫 SHA256 `0dce8edcf757a697865899731c2ea68ddaffeae553ceb0c7181008f08f961391`。設定維持官方 scorer、seed 0、repeats 1、trials 1、concurrency 1、tolerance 0.05、min_n 5、bootstrap 1000 與 min_coverage 0.95；未開啟 minimization。官方 SDK 接收完整原始人口，沒有縮樣或將不同評測合併。進入原生模型時的型別轉接會在 312 筆請求中省略合計 927 個 `criteria: null` 欄位，另有 24 筆只改變題目欄位排列；獨立盤點未發現 head 或候選順序變更。此處記錄的是轉接差異，不把不同序列化視為原始位元組完全相同。

## 獨立重算與來源範圍

獨立稽核以本次捕捉的 1,248 筆結果作為唯一回應來源，建立新的 `:memory:` 快取，重新執行固定官方 SDK；沒有呼叫模型。原請求順序、原生轉接與正規化、240 個 base、1,200 個試驗、1,468 筆 outcome、完整分數、coverage 與 1,000 次 bootstrap 結果均與原報告一致。稽核使用對 JSON 型別及對映順序敏感的比較，並核對 4,992 筆分階段事件、1,248 對 GPU 租約及每次 callback 的時間紀錄。

稽核重新驗證 51 個儲存庫公開程式與設定的來源 pin，並核對其受控 release 與 held source 的 gate 關係；另有 7 個外部檔案不在本次獨立讀取範圍，包含 Python 執行檔、torch 及模型／overlay 來源。worker 自身的來源及模型檔案檢查不等同本次稽核重新讀取這些外部檔案，也不證明執行中每個 GPU backbone 張量的內容。

外層監督紀錄包含 762 次主機取樣，取樣時間涵蓋全部 callback 的起訖；各次觀察的 ComfyUI running／pending 均為零，可用記憶體均達基本門檻。最長相鄰取樣間隔約 0.528 秒，這些離散觀察不證明間隙內狀態。監督器選用的較高記憶體門檻未記入關閉紀錄，無法由該紀錄獨立核對。worker 結束時回報其 CUDA allocated bytes 為零，外層紀錄顯示 cgroup 已消失；這些不代表整台機器的 GPU 或共享記憶體歸零。

## 時間與解讀限制

1,248 筆完整受保護 callback 的 p50／p95 為 **0.286／0.431 秒**，採 `(n−1)×q` 線性插值。計時從 callback 入口至回應完成核對，包含來源與狀態 guard、通用事件留證及同步；不包含模型載入、時間日誌自身寫入及 SDK 快取命中，也不是單純 GPU forward 時間。本次背景 CPU pytest 工作是否與評測重疊未能由現有時間證據確定，因此不把延遲視為乾淨或公平的速度比較；不得與[先前分段評測](../2026-10-25/decision3-native-coherence-gpu-results.md)僅 327 筆新增 callback 的分位數混合。每次成本與耗電成本未量測。

這是公開、非 sealed 的一致性研究測試，沒有官方榜單排名或競品勝出結論；公開資料可能已暴露於模型訓練或過往調整。它不衡量網頁導航或真實任務完成，也沒有驗證瀏覽器增量頭的效果。Benchmark Heaven 的準確率與校準結果另見[本次 public231 報告](decision3-dual-head-heaven-public231.md)，不與 coherence 分數合併。

獨立稽核程式 SHA256 `3dce49447fa0a5ed9f6a84d36c77cb8486e7102829223dac6386cc72b05073ff`；完整聚合 stdout（含結尾換行）SHA256 `404a3775ed3d45526d700723063cc259198e36c5cd71695dff86c36004681cd3`。在有本次原始證據及固定公開 checkout 的環境中，可從儲存庫根目錄重算：

```sh
env -u PYTHONPATH .venv/bin/python -B -s scripts/jevbench_coherence_scorer_audit.py \
  --run-root "$EVIDENCE_ROOT" --official-root "$PUBLIC_COHERENCE_CHECKOUT"
```

`EVIDENCE_ROOT` 指向本次執行證據，`PUBLIC_COHERENCE_CHECKOUT` 指向上述固定官方版本。稽核只輸出聚合結果，不執行模型；原始回應、逐題內容及本機證據不包含在此文件或公開 Git 提交中。聚合 stdout 若重新格式化，其雜湊會改變。
