# Jev-Omni 差距診斷與下一輪實驗

完整 231 題中，S1 independent 為 **197/231（85.28%）**，Jev-Omni 為
**203/231（87.88%）**。淨差六題：13 題只有 Jev 對，7 題只有 S1 對；
配對 source-group bootstrap 的差距 95% CI 為 −1.29 至 +6.72 個百分點。
這是公開開發題的觀察差距，尚未證明正式排名或全面能力差距。

完整拆分、原始資料與 hashes 見 [錯誤診斷](error-analysis.md)；
舊 GPU 結果及 replay 稽核見 [既有 campaign](../jevbench/README.md)。

| 項目 | S1 | Jev-Omni | 含義 |
| --- | ---: | ---: | --- |
| Noul | 62/74 | 66/74 | S1 在 39 個 negative cases 中錯誤接受 11 個，Jev 為 6 個 |
| Choice | 120/139 | 123/139 | 差距包含長政策、條件與描述綁定 |
| Score | 15/18 | 14/18 | S1 的 expected-value MAE 也較低；需要保留 |
| Strict coherence | 71.96% | 74.55% | 完整五個 dimensions，不等於任務 accuracy |
| MEA | 37.99% | 49.26% | 跨型別、互補與累積機率仍有改善空間 |
| LOG | 85.42% | 75.42% | S1 的優勢，不能為改善其他項目而忽略 |

目前可核對的訓練差異是：S1 已發布版本實際只使用 2,048 個 BoolQ
Noul 訓練案例，沒有 Choice／Score training cases；Jev-Omni 的
[固定版本 model card](https://huggingface.co/akhilaaa3/Jev-Omni/blob/5addda86ddee081a68fb067477ea100c221b8917/README.md)
聲明使用 30,000 個問題微調。訓練覆盖差異是可測假說，不能由兩個模型的分數直接證明因果。
另一個機制差異是題目放在 USER 或 ASSISTANT 回合；已固定對照來分離提示位置與權重影響。
正的 temperature 不改 argmax，單獨調溫度無法修復分類錯題。

[執行前協定](protocol.md) 固定八個 profiles、128-update 配方，以及獨立的
1,024 個 train／192 個 calibration／384 個保留模板 test cases。
Head 與 LoRA 都不使用公開題或 coherence 題來 fitting；每個 profile 都須保留
完整 BoolQ256、native media52、historic text128、native fixture8/29 與官方測試人口。
合成測試衡量有限世界的機率與模板遷移，不能替代真實世界或 sealed benchmark。

截至執行前檢查完成：來源與協定保存於本機 commit `cc20b68`；
新工具的針對性測試 **109 passed**，完整 repository suite **1,349 passed、3 skipped**。
Skips 是未安裝的 SentenceTransformer 與 bitsandbytes；另有一個既有 Starlette deprecation warning。

**本輪八組 A100 實驗尚未啟動。** 自動核准審查拒絕啟動 Modal，理由是需要使用者明確授權
外部雲端資料傳輸與 A100 費用；已提出兩階段的具體授權範圍，等待回覆。
所有新 profiles 仍為 `not_run`，沒有新訓練權重、性能改善、成本或上榜結論。
既有發布模型與 SDK 預設保持原狀。

另外執行獨立的本機離線檢查，與上述八組研究人口分開。真實 Gemma processor
先揭露 text-block trimming 會去掉 state 與問題之間的換行；已修正研究 helper
使 text-only 的兩段保留在同一 text block，沒有改 SDK 預設。
本機 MPS probe 只對固定 native fixture 8 cases／29 questions 做同權重 prompt 對照，
不訓練、不 fitting、不下載，不作正式品質或速度結論。
首個 probe 在載入模型前被 MPS `precision` 參數檢查拒絕，失敗 receipt 保留；
改用 SDK 支援的 `dtype=torch.bfloat16` 後以新目錄重新執行。

本機 MPS 第二次執行已完整完成：**current 29/29、user_question 29/29**，
全部 8 cases、text／image／audio／mixed 都納入；0 label flips、29 distributions 改變，
最大 absolute probability delta 為 `0.13401752710342407`。兩組都是逐題 native forward，
同一模型與裝置、固定原 release temperature；沒有用這些結果選 prompt 或改訓練配方。
58 份答案由原始 logits 重算完全一致，62 份執行 source 的 SHA256 全部驗證；
實際 config／權重 hashes 與保留的公開發布 manifest 一致。
完整 [paired audit](../../../../artifacts/jevbench/breakthrough-20261008/local-mps-02/paired-audit.json)
及 [CPU processor check](../../../../artifacts/jevbench/breakthrough-20261008/local-processor-check.json)
保留原始支持。這只證明此 fixture 上能正常推論與 labels 保留；
機率明顯變化，不能聲稱 distribution parity、改善 calibration、泛化或速度提升。

全部新工具完成後，最後完整 suite 為 **1,425 passed、3 skipped**，Ruff check／format
通過。新的 [outcome analyzer](../../../../scripts/analyze_breakthrough.py) 會獨立重算完整人口，
目前的 [A100 not-run audit](../../../../artifacts/jevbench/breakthrough-20261008/analysis-not-run.md)
明列八個未執行 profiles；不把本機 smoke 或歷史模型成績補入這些人口。

[完整證據 ZIP](raw-evidence.zip) 共 235 members、5,231,356 bytes；
SHA256 `9358fa7bccf42805eb0f326ee85258801cfa4e96688655e7f3173d6a69bce0e0`。
[Manifest](raw-evidence.manifest.json) 保存每個 member 的 SHA256、size、CRC 與執行範圍，
其 SHA256 為 `751da55a153ae01d7ee0eabc3447145cd1912ca2848ef41ad3462ffa987d0491`。
獨立逐檔驗證通過；含 frozen protocol/data、原始 executed sources、processor 修正前後、
失敗與成功的 MPS probes、paired audit、outcome analyzer 與發布 metadata。
沒有任何新訓練 head／adapter；baseline model weights 依 immutable identity 另取得。

```bash
uv run --no-sync python scripts/archive_breakthrough.py \
  --verify docs/validation/2026-10-08/breakthrough/raw-evidence.zip
```
