# Decision3 text-history：原生瀏覽器資料收集

日期：2026-10-27

使用 jev-ultrafast 的真實瀏覽器執行收集了 **33 個 TRAIN 場景、120 筆資料**。收集使用腳本教師，沒有呼叫模型；獨立 admission 與原始受限程序的關閉驗證均已通過。這些資料用於後續訓練研究，不能作為模型表現分數。

每筆原生請求的操作選項數均為 K7，且各有 4 個 heads；全批合計 480 個 heads、4,882 個 legal positions。教師操作分布如下：

| 操作 | 筆數 |
| --- | ---: |
| CLICK | 45 |
| TYPE_TEXT | 30 |
| SELECT | 12 |
| DONE | 33 |

逐步核對中，可見欄位值、type head 的目前值與 DOM 值相符，發布的近期操作歷史與原生歷史相符，各為 **120/120**。另確認 57 筆在同一欄位成功輸入後，當前值仍正確且教師選擇非 TYPE 操作；6 筆則在同一欄位成功輸入後，控制項清空或改寫該值，教師再次選擇 TYPE，其中觸發控制項為 CLICK 3 筆、SELECT 3 筆。這些是教師軌跡的資料覆蓋，不是模型決策結果。

獨立收集審核 SHA256 為 `0a703e8c78b27cabc3eb9eaa6c8bdbf7f62f129ba95f62b21173e9440621def6`；欄位值與歷史核對 SHA256 為 `2595f8088423ec9408688827d5281b782b7284a0186a0afaa37976f84ca05870`；同欄位與重設反例核對 SHA256 為 `a01b8a1d24cb2d957039966741fefdbaa06f99eb8f17f20c7e87306242abbcfd`。特徵輸入驗收 SHA256 為 `06d41294b75e4b56f643850515fdc4e4524f5c4959310377972c2d61523f72c3`，其中 `fit_authorized=false`。

本次 `model_calls=0`，沒有模型分數，也沒有新的 JevBench 或完整瀏覽器評測分數。先前的 [首案例配對](decision3-scroll-state-firstcase-pair.md)中，candidate 與 baseline 各為 0/1；candidate 重複輸入欄位中已有的文字。這次資料收集尚不能證明該問題已改善或模型勝過其他模型，後續仍需 GPU 特徵擷取、訓練，以及未見過的真實瀏覽器案例測試。收集與驗收流程見 [訓練資料收集指南](../../decision3-affordance-training-collection.md)。
