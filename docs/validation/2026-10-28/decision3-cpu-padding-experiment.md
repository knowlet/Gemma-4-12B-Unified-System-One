# Decision3 CPU 長輸入分組實驗準備

先前 GX10 多核心診斷的四個 head 長度為 3117、4196、4579、3126 tokens，10 核原生呼叫約 23.3 秒，未通過 20 秒期限。這次加入預設不可啟用的文字分組候選及獨立 CPU 監督器，準備在五核心上以接近同長度的自造輸入比較 publisher 原生 `system_one`、不分組的 prepared 路徑、原順序兩題分組與按長度兩題分組。每種模式計入完整請求時間；候選時間還包括資料快照與驗證開銷。

按精確目標長度計算，不分組、原順序分組、長度分組分別需要 18316、17550、15410 個 padding slots；這只是一項幾何估算，不代表運算時間同比下降。現有 CPU 實驗與兩套 JevBench owner 的合成整合測試共 116 項通過，未載入真模型。實驗會保留各次完整機率與選擇，並比較不同模式結果；來源身分、實際資源、結束狀態和輸出仍須另行驗證。

[Decision3 官方 batching 紀錄](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/decision3-batching.md)指出不同批次組合可能因數值捨入改變答案；該紀錄使用不同硬體和路徑，不能套用為 GX10 的測量。此候選與正式後端分開，沒有證明數值等價、速度改善、真實網站任務成功或 JevBench 表現。
