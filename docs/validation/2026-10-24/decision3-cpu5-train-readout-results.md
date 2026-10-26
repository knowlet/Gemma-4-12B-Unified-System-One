# Decision3 CPU5：96 個 train 狀態與首次真正讀出層訓練

GX10 已完成12個獨立 train 流程、96個狀態的真正模型特徵收集，逐組 root 驗收後，用固定32步 SGD 訓練獨立 FP32 delta。候選已在任何 dev／sealed／JevBench／瀏覽器評測前凍結。這是訓練完成紀錄，尚無品質提升或勝出證據。

| 項目 | 結果 |
|---|---:|
| train 狀態／流程／家族 | 96／12／4 |
| 真正模型載入／原生決策呼叫 | 12／96 |
| 同次 forward 特徵列 | 384；每例4×1024 FP32 |
| 有效監督目標／略過未標註目標 | 132／252 |
| W0／delta | 255×1024 FP32，原始 W0 保留 |
| 收集總 root 時間 | 943.461 秒 |
| 完整單次收集交易範圍 | 7.787016～8.262311 秒，均低於20秒 |
| 32步訓練 root／child | 2.841495／1.179731 秒 |
| 訓練 owned unit 峰值 | 47,239,168 bytes |
| delta 非零元素 | 8,192；第8～254列維持0 |

每組收集都保留原 CPU15–19／quota500%／4GiB／swap0／call20／whole120 契約，並經本次原始 root0、owner完成與 scoped closure 驗證。Root 核對全部特徵、W0、原始分布與選項順序；保留 producer FP32 Torch／NumPy 混合容差驗證，另以標準函式庫逐合法槽位重播。其他審閱者只讀來源，沒有第二位實測資料審閱者。

訓練只讀3個 train exports與12份 root COMMITTED sidecar所固定的資料。沒有把數值診斷陣列、官方評測答案、dev／sealed或已曝光的 Copper 場景送入訓練。這是程式的檔案／流程隔離，沒有宣稱 OS 存取封鎖。原始模型、特徵、標籤內容、原始回答和delta陣列均留在本機忽略目錄。

預先固定 steps32、learning_rate0.001、L2=0.0001、seed20261023、batch64。小批次 loss 首次0.984286、末次1.237235；兩次採樣不同，不能當作完整 train loss 的前後比較，也沒有據此宣稱學習有效。候選別名為 `d3-edge-dsl01-train-delta-cpu5-candidate`，delta SHA為 `812256139b35429dc55f4f178af8c177d2a31c023414e668e9b39cdc217d14be`。後續推論須保持 `hW0ᵀ+hΔᵀ` 的計算順序，不合併 W0 與 delta。

保留兩次實際修正紀錄：收集進度檔沿用載入器的階段字串，root 驗收改為核對真正 completed／逐筆與原始執行證據，未改產物；fit18 在進入數值計算前因子程序 systemctl查詢失敗，fit19沿用相同核心／程序限制直接讀回，父監督程序仍執行原 systemd查核。所有原失敗與已用來源保留。collector14、held loader／fit13、released fit14、closed audit5個來源測試通過。

兩套 JevBench 分開列示且分數未更新：原生 Decision3 準確度 **158/231（68.40%）**，Gemma **195/231（84.42%）**；機率一致性 Decision3 **1234/1248未完成，不報分**，Gemma **41.26/100**。先前真正 HTTP＋Chromium 完成狀態判斷仍是失敗紀錄。下一步是候選推論的零差異／非零差異驗證，再進行 dev／sealed與瀏覽器前後比較；目前沒有新場景通過宣稱。

詳細彙總見 [JSON](decision3-cpu5-train-readout-results.json)。
