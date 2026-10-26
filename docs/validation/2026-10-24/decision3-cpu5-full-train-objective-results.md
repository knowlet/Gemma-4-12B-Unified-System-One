# 凍結讀出層：同一完整 train 集的前後比較

在固定32步訓練及候選凍結後，GX10以新 owner 執行唯讀診斷，對相同96個 train 狀態中的132個有效監督目標比較原模型與候選。沒有再訓練、改參數或讀取 dev／sealed／官方題目。

| 完整132個 train 目標 | 原模型（zero delta） | 凍結候選 |
|---|---:|---:|
| 合法目標命中數 | 80／132 | 89／132 |
| 命中率 | 60.6061% | 67.4242% |
| 完整平均 soft CE | 1.1017837524 | 1.0410724878 |
| 同一L2項後完整目標值 | 1.1017837524 | 1.0410726070 |

命中率提升6.8182個百分點。多個可接受目標均視為有效，tie沿原選項順序，252個未標註目標仍略過。這證明該候選在這份訓練資料的完整目標上較好，**不是未見場景、瀏覽器或 JevBench 的測試成績**；不能據此宣稱打敗其他模型。

本次 root 執行2.934133秒，原始 child交易低於120秒，正常 root0／owner完成／current scoped closure；40筆來源身分核對通過。輸入保持唯讀，沒有 optimizer、梯度更新或模型 forward，使用既有同次 forward 特徵。Root 逐筆重建全部 features／W0／targets／counts／delta 位元組雜湊，再以純標準函式庫的 float64 累加重算合法 argmax與 soft CE：命中數同為80、89；與原FP32結果的CE差距僅9.02e-8、3.39e-8。這是額外重播，不冒稱 root 重跑原FP32 runtime；其他審閱者只讀來源。

候選推論 composition 仍為 SOURCE_ONLY，12個來源測試通過，已補必要紀錄寫入失敗拒絕回傳，以及先移除hooks再複製觀測紀錄的回歸測試。下一步須用真正模型驗證zero／nonzero delta、4與2個真實問題列，再接候選adapter／HTTP／瀏覽器。

兩套 JevBench分數未變：先前原生Decision3準確度158/231（68.40%），機率一致性只完成1234/1248而不報分；Gemma分別為195/231（84.42%）及41.26/100。這份train診斷分開記錄，不列入任何一套JeVBench分數。

詳細證據見 [JSON](decision3-cpu5-full-train-objective-results.json)。
