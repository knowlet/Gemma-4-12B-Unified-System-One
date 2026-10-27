# 跨文件循環提示：合成驗證與開發資料重播

日期：2026-10-27（Asia/Taipei）。本次加入預設關閉的跨文件循環提示；它只提供觀察摘要，不修改候選選項、模型答案或任務裁判。正式接入仍需要明確啟用與獨立審查，不代表真實瀏覽器效果已驗收。

啟用後，提示只使用同一 episode 中經 owner 確認執行的動作與前後完整觀察；動作目標須能以不依賴文件內節點序號的語意資料唯一辨識。比較保留完整 URL、可見文字、控制狀態與原生控制順序，並檢查文件身分、連續快照與下一次模型呼叫的 freshness；缺少、歧義或不一致時回傳 unknown 並中斷鏈結。這仍只是受限的可見投影，不是完整 DOM 或任務成功證明。

以先前已參與選型的單一 DEV 瀏覽器配對保存資料進行離線重建，結果如下。此資料已用於開發，**不是未見測試**：

| 重建範圍 | 原同文件模式 | 跨文件 opt-in 模式 |
| --- | --- | --- |
| 候選臂 | 無循環或重複訊號 | 15 次 known、1 次 unknown；7 次重複轉移、4 次交替循環，兩類可重疊 |
| 候選臂下一次 policy state 對帳 | 無提示 | 6 次產生提示；另一次訊號因 state 綁定不符而未發送 |
| 基準臂 | 無循環或重複訊號 | 5 次 known，無循環或重複訊號 |

此重播由保存快照重建 execution 與 freshness attestation，並非即時 owner 直接作出的來源證明；6 次是離線重播中 policy state 對帳通過的提示，不證明它們曾傳給真實模型。沒有重新執行瀏覽器、模型或裁判，也沒有衡量成功率、泛化、速度、成本或正式排行榜分數。單一配對的限制與原獨立 DOM 判定見[先前紀錄](decision3-link-navigation-firstcase-pair.md)。

執行四個相關合成測試模組共 **114 項通過**，涵蓋原有預設行為、跨文件往返、控制或頁面變化、歧義目標、episode／文件錯綁、未確認執行與 stale freshness；Ruff 與格式檢查通過。另有獨立 review 執行三個核心模組的 **75 項測試通過**。這些測試不使用模型或真實瀏覽器，不取代端對端驗收。

原始開發快照不納入 Git。保留完整本機證據及相同來源的環境可從儲存庫根目錄重播；輸出路徑必須是不存在的新檔案：

```sh
env -u PYTHONPATH .venv/bin/python -B -s \
  artifacts/jev-runtime/decision3-text-history-combined-fit-cpu5-20261027-01/root-link-pair-cross-progress-replay-v1.py \
  --output "$NEW_ABSOLUTE_OUTPUT"
```

本機 replay 程式 SHA256 `5a45a0863b35b01dd0f62fee2965d68252b981bee7a8455746419cba7aaad869`；兩次彙總輸出 SHA256 均為 `36f5bbb8ecda75edf0fa00b385ed6fc23b67241bae8d16fd0d96a53c1e4284e1`。這是同一保存輸入的重播一致性，不是兩次獨立模型實驗。
