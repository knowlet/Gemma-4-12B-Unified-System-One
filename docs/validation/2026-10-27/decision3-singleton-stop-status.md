# Decision3：singleton DONE 覆核終止狀態

日期：2026-10-27（Asia/Taipei）。DONE review owner 新增預設關閉的 `singleton_stop_enabled`；它仍是 `SOURCE_ONLY`，不會自動啟用。

此分支只在原始 fresh 決策為 DONE、覆核為 unsupported 或 unknown、原始 operation 選項**恰為 `{DONE, BLOCKED}`**，且當前 action space 沒有 targets 或 controls 時，才在再次檢查同一頁面、決策與狀態後由 owner 本地設為 blocked。它消費待執行決策並保留原 DONE 決策紀錄，不再呼叫模型、偽造 BLOCKED 的模型選擇或機率，也不改原 judge。blocked 是安全終止，不能當成任務成功；覆核結論本身亦不證明真實目標是否完成。

合成回歸涵蓋兩種覆核結果、功能關閉與非 singleton 路徑、呼叫預算、新鮮度、事件 callback 重入及狀態變造、計時異常與 snapshot 失敗。相關三個測試模組共 **71 項通過**；擴大至 owner、parser、binding、child、launcher 及對帳的混合收集為 **160 項通過**。測試包含真 native parser、service handler 與 semantic binding，後端和瀏覽器皆為合成替身；這不是 GPU 數值或真實網站驗證。

另以公開 upstream 固定版本 [`1231850`](https://github.com/browser-use/jev-ultrafast/blob/1231850a0bf1a0c0341fe408ef1668dbbfdfac46/jev_ultrafast/agent.py) 的原始 Agent 進行一次性記憶體合成探針；來源 SHA256 為 `aaf8b271cb0ee6643fe7beb525328ab80de8568c7ac044d02b9c6721161e7c8b`。以假 browser、model 和 transport 接上本版 owner 時，unsupported 與 unknown 均保留原 DONE 決策紀錄、執行兩次 policy 呼叫、以 blocked 結束，未送出瀏覽器輸入；終止後的重播被拒絕。探針不是持久執行收據，也不驗證真實運行時的來源。

新事件 `singleton_stop_requested` 只表示提出本地終止；既有 reconciliation 尚未將它與原決策、同次覆核、最終狀態及獨立 judge 證據完整綁定，會保留 **unresolved**。事件本身不能證明最終狀態或判分。現有 release 沒有包含這次來源變更；尚未建立對應 release，也沒有使用真 GPU 或瀏覽器測試此分支。

本次受審 owner SHA256 為 `dcca02da3a2c37432c32ecd2c379b36be847ed47aac983c6d41231fe9178fe29`。上述驗證不代表模型品質改善或可上線資格。
