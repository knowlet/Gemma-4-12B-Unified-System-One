# GX10 Decision3 真實 HTTP 與瀏覽器診斷

CPU5設定已真正串起原版Jev caller、模型、HTTP及Chromium；兩個按鈕的診斷頁沒有通過。這次失敗已有完整傳輸與原生推論證據，問題集中在完成狀態的行動選擇。

模型先正確點選銅色按鈕，下一次輸入的`state.page.text`已包含「Copper lamp armed」，卻繼續點銀色、再點回銅色，最後選`BLOCKED`。實際序列為`CLICK → CLICK → CLICK → BLOCKED`，三次均為真正的trusted DOM點擊。`DONE`和`BLOCKED`始終完整提供，沒有過濾選項、改寫答案、fallback或文字helper。

四次HTTP均為200，客戶端觀測時間約1.73、1.53、1.68、1.89秒；每次都是原生batch4執行中的兩列tail，實際形狀依序為[2,578,1024]、[2,545,1024]、[2,569,1024]、[2,595,1024]，沒有補假問題。只載入一次原模型，保留BF16 backbone／原FP32 head、temperature1、context16384及原20秒cooperative gate、25秒caller設定與16 ticks。未宣稱硬取消或網路送達時限。

主程序在結束後逐筆核對四個實際request ID、雙向body hash、完整原生機率與argmax、未修改的request／candidate／response，以及62個選定來源、20個model／28個runtime紀錄與原始正向authenticated_stop清理。來源與原upstream版本不變。原caller判定失敗，因此root為1、owner.completed=false；這些結果保留原狀。worker在caller丟錯後尚未自動載入report做HTTP閉合，四筆閉合由主程序事後審查完成，沒有把worker失敗升格成成功。

模型與Chromium共用4GiB／swap0的原unit峰值為1912430592 bytes，觀測的memory.events全為0。這僅驗證此診斷的合併用量，不推論長網頁或整套benchmark容量。

第一次啟動另暴露進度紀錄錯誤，在載入模型前因重複使用exclusive writer停止。新worker保留編號快照，再以既有helper原子更新latest；10項檢查通過後重跑。失敗來源、spec、freeze及原件均保留，未修改原boundary/helper。未增加斷電耐久性宣稱。

這是一個已曝光的人工診斷頁，不是正式瀏覽器分數，也不拿它訓練。後續以隔離的train／dev／sealed資料改善完成狀態判斷，再回到真實網頁驗證。兩套JevBench分數仍分開：先前Benchmark Heaven public231為68.40%；Decision3的coherence MINI完整評測尚未完成，因此無合格分數。均勻機率控制的89.41分是指標診斷，並非模型能力。

精確數值與證據SHA見[JSON](decision3-cpu5-real-browser-diagnostic.json)。本報告為主程序審查；第二位審查者僅讀來源。原始狀態、回應、機率、特徵與權重不公開，尚未優於其他模型。
