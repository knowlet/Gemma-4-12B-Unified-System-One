# Decision3：跨文件連結訓練資料收集

日期：2026-10-27

接續[瀏覽器短測](decision3-link-navigation-browser-probe.md)，使用固定版本的 jev-ultrafast、settled V7、Chromium 與腳本教師，在 GX10 真實瀏覽器收集 27 個獨立的 train-only 多文件場景。每個場景均完成，瀏覽器已關閉，沒有呼叫模型。資料採集程序限制為 CPU 15–19、500% CPU、8 GiB 記憶體及零 swap。

共收集 72 筆原生請求：45 筆 CLICK 和 27 筆 DONE。CLICK 的實際點擊目標選項數分別為 1 個的 18 筆、8 個的 9 筆、16 個的 9 筆、24 個的 9 筆；1 個選項的資料並不能證明模型能從多個目標中選對。每次導航與完成動作都由原生瀏覽器執行，教師標籤不是模型預測。

另一個受限程序使用[批次管線](../../../scripts/jev_decision3_link_navigation_pipeline.py)重新驗證固定來源、排除清單、瀏覽器網路事件、CDP session、原生 publisher、動作、跨文件轉場和每筆標籤及遮罩。重播通過後產生原始 wire 投影；這使用與收集器共用的驗證函式，並非完全不同實作。投影仍為 `fit_eligible=false`，訓練前另需核對資料合併與模型特徵。排除清單只涵蓋已核對的目標與身分，不能保證所有未揭露評測均無交疊。

計畫 SHA256 為 `1fbe2ea9f890691b84d394b233725e1cf3ac196334aeaba03538f290cf390976`，收集收據為 `ca4f85ef728a4ec2b785bd890114821927af50bb6f0ba5a1f6b2acba14837634`，重播紀錄為 `521e36da2d71f95daef309d9d2cd8b19c09518840b9870ac91536aec8b9b4ce8`，待審核投影為 `53034ffc2c4b3f35ddae85fd0b7b1bebc2f668ebc902bad02a9e261f81aad561`。

這些數字描述教師軌跡與選單覆蓋，不是模型效能、瀏覽器成功率或 JevBench 分數。尚須擷取模型特徵、訓練、獨立驗證及真實任務比較。
