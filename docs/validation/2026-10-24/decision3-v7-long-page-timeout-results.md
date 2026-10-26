# Matched V7：長頁面原生逾時，完整比較尚未完成

GX10 的這次 fresh-browser campaign 使用測量來源提交
`cefdf318dcb34a80ac5f029dbc002b2f8ef25930`。Gemma 基準完成全部十案，
獨立驗證成功 **4/10**：known4 為 **1/4**，public6 為 **3/6**。
78 次 policy 與 6 次 helper 呼叫共 84 次，HTTP 回應全部為 200；
四案通過 fresh DOM 的 DONE 驗證，其餘六案保留失敗。

| Gemma 十案結果 | 案數 |
| --- | ---: |
| 通過獨立驗證的 DONE | 4 |
| 宣告 DONE，但獨立驗證失敗 | 2 |
| 三次無可觀察頁面變化後自動 blocked | 2 |
| runner 的 16 次決策預算耗盡 | 2 |

前一次 matched-02 的相同模型為 3/10；兩次之間沒有模型更新，網站狀態
會變動。這次多成功一案尚無模型改善證據，也未建立穩定提升。

CPU5 的 Decision3 operation 候選模型只完成前兩案，兩案均由 policy 選擇
BLOCKED、HTTP 200，成功 **0/2**。這是已完成前綴的結果；候選模型的
完整十案分數與完整 matched 比較分數均維持 **null**。

第三案的原生 batch shape 為 `4 × 4579 × 1024`，四個輸入 token 長度為
3117／4196／4579／3126，合法選項數為 7／38／36／1；logical token 共
15,018，padding 後共 18,316。原生 forward 在 **30.438 秒**
返回，隨後原本的 **20 秒合作式期限檢查**拒絕回傳，client 得到
HTTP **504**，未授權瀏覽器動作。

整次 campaign 為 **615.499 秒、exit 1、owner_error**。
原始中止原因是第三案的原生期限失敗；沒有觸發 ComfyUI 或記憶體中止。
三次 CPU 執行的原 scoped closure 都為正向，各有兩次 clear scans，
選定來源在原始收尾時保持不變，沒有本次模型或瀏覽器程序留下。
actual32、actual33 的 `completed` 與 `service_success` 為 true；
actual34 的兩個失敗值 false 均保留，與正向清理證據分別列出。

兩套 JevBench 與這次瀏覽器結果分開報告：

| 評測 | 未適配的 Decision3 Edge 原模型 | 本次 operation 候選模型 |
| --- | --- | --- |
| Benchmark Heaven public231 | 158/231（68.398268%） | null，尚未評測 |
| JevBench coherence MINI | 70.58660130718954/100；1,248/1,248 請求完整 | null，尚未評測 |
| 本次 fresh V7 瀏覽器 | 本輪未選用 | 已完成前綴 0/2；完整十案分數 null |

原模型兩套分數的[完整結果](decision3-edge-cpu4-coherence-complete-results.md)
另有原生回傳與離線重算核對。這些分數不套用到新的 operation 候選模型。
這次尚未完成模型間的十案對照，也沒有已驗證的競品勝出結果。公開內容只保留輸入形狀、
計數、時間及原生收尾等彙總；沒有加入原始請求、回應、DOM、gold、機率
或模型檔。後續 CI 的來源修改也不歸入本次測量來源。

下一步以相同自造長輸入、相同精度與資源限制比較 GX10 CPU5／CPU16，
先量測長上下文 forward 是否能符合原 20 秒期限。後續完整瀏覽器評測
仍保留所有失敗與原本分母，另行驗證任務成功率。

本報告的[機器可讀彙總](decision3-v7-long-page-timeout-results.json)保留原測量提交、計畫及獨立核對來源的 SHA256。原始私有內容未發佈；這份彙總不聲稱完整公開重播或完整候選輸出清單驗收。
