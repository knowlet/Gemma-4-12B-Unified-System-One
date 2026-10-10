# JevBench 研究發布與 CPU 重播補充

日期：2026-10-10（Asia/Taipei）。[研究預覽版與下載入口](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/tag/research-2026-10-10-jevbench)發布本輪工具、結果與原始證據。這份補充記錄 10 月 9 日封存之後的實際授權、CPU 回傳修補與校驗結果；先前 ZIP、來源及報告保持原始位元組。

- [本日完整結果](cpu-replay.md)：八組／兩種政策共 19,968 次回答重播，529,136 個機率值及 native answer fields 全部一致；兩組校準 metadata 仍有微小差異，完整稽核維持六組通過、兩組未完成。
- [下載檔與 ZIP 內路徑](evidence-layout.md)：公開 assets、逐檔 manifest、舊本機 `artifacts/…` 連結對應的封存位置。
- [原始八組研究報告](../2026-10-08/breakthrough/results.md)：公開題、合成 test384、校準、媒體、歷史文字、訓練與 Jev-Omni 的完整比較。
- [市面模型 campaign](../2026-10-08/jevbench/README.md)：OneJev、Decider、Jev-Omni 與 S1 的既有完整公開人口比較。
- [歷史封存重組](../2026-10-08/breakthrough/evidence-download.md)：兩份 Git 分片可重組完整八組實驗 ZIP。
- [所有日期的驗證索引](../README.md)：依日期與用途查找發布、模型比較、公開研究及重播證據。

公開研究人口仍為 197→200/231，歷史 Jev-Omni 為 203/231，配對區間包含零。Head／LoRA 候選的校準後與媒體保留表現不足以採用；既有正式模型維持原版本。sealed evaluation 未執行、排行榜未提交，formal acceptance／rank／完整費用仍未知。

發布前完整本機驗證 **1,712 passed、3 skipped、1 個既有 warning**；Ruff、格式與 diff check 通過。輸入快照上傳與 CPU04 作業已完成，不把模型品質接受、跨平台逐位元一致或正式上榜混為同一結論。

本機整理範圍限於完整重複 ZIP（116,641,459 bytes／約 111MiB）：確認發布檔可下載且雜湊相符後，可由兩份 Git 分片逐位元重組。模型檔、完整原始人口、原失敗診斷與歷史封存保留；實際發布核對與整理收據另隨最終紀錄保存。
