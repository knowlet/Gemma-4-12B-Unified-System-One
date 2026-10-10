# Jev-Omni 差距與八組優化實驗

2026-10-10 僅補正公開下載連結；封存原始版本及當時數值不改寫，後續 CPU 證據見 [10/10 索引](../../2026-10-10/README.md)。

兩階段 A100 的八組固定實驗已完成，原始資料已全部取回。獨立稽核有六組完整通過，兩個訓練候選的跨平台機率連結校驗仍未完成；目前不採用新訓練權重，不改 SDK 預設。完整數字、支持人口、配對區間及校驗狀態見 [結果報告](results.md)。

公開 231 題的原版為 **197/231（85.28%）**，把問題放在 USER 回合的未訓練對照為 **200/231（86.58%）**；歷史 Jev-Omni 為 **203/231（87.88%）**。USER 對照與 Jev 的配對差距是 −1.30 個百分點，95% source-group 區間為 −4.93 至 +2.58 個百分點。這是已公開、已觀察的研究題目，不能由三題差距推出正式能力高下或榜單名次。

原版主要差距包括 Noul 錯誤接受：39 個 negative cases 中錯 11 個，Jev 錯 6 個；USER 對照減至 7 個，但原生媒體由 33/52 降至 31/52，historic text 由 118/128 降至 117/128。提示位置有改善訊號，也有保留能力的代價；本輪沒有依公開分數挑選或發布 checkpoint。

訓練也有正面訊號。LoRA 在未校準的合成 test384 上，把 Brier 降低 0.324543，95% 配對區間為 −0.351031 至 −0.299728；但每個候選各自使用獨立 calibration192 校準後，LoRA 的 Brier 為 0.103111，對照是 0.068526，差值 +0.034585 的區間為 +0.027448 至 +0.042025。Score expected-value MAE 也退步；影像／音訊 30/52、BoolQ 229/256，低於對照的 31/52、231/256。改善原始機率與達到整體採用條件是不同結論。

另一個具體缺口是訓練覆蓋：新 train1024／calibration192／test384 全是文字，Choice 候選為 2／4／8 個；原生 media52 全是十一個候選，其中 digit-8、digit-9、unknown 沒有直接候選位置 loss exposure。十題的 gold 在 digit-8／9；runtime 本身支援這些候選。完整52題分拆顯示：head-init→head 在 gold0–7 的42題維持27對（3個新錯／3個修正），gold8／9的十題則由4對降至0對；LoRA唯一新增錯誤是 gold0–7 的音訊題，gold8／9仍4對。覆蓋缺口是 head 的可驗證假說，不能解釋 LoRA 的全部退步或直接證明因果；[完整分拆（ZIP：research/final-coverage-diagnostic.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip) 保留 source groups、錯題與 tensor 支持。

下一輪的研究方向是補足獨立原生媒體訓練及十一候選位置、保留三種題型與原版能力的混合訓練，並使用貼近部署人口的獨立校準。新資料需避開已用的公開題、coherence 題及保留測試，再先固定對照與驗收條件；本輪 frozen protocol、配方、資料與 checkpoint 均不因此改寫。具體比較與限制列在 [結果報告](results.md)。

[PR #6](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/pull/6) 已於 2026-10-08 合併，上游 main 已對齊；相關 contracts（Python 3.11／3.12／3.13）與 inference CI 成功。合併 metadata 與 [專案來源稽核（ZIP：research/project-update-audit.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip) 保留完整 commit 與 tracked-tree 核對。這不等於本輪研究分支已推送或取得新的 CI。

市面公開發布模型的既有完整比較包括 OneJev、Decider、Jev-Omni 與 S1 variants，見 [JevBench campaign](../jevbench/README.md)。不同模型的 calibration／runtime／硬體及公開曝光都保留；只有 registry 設定而未執行的模型不補成實測。本輪八組是固定優化對照，不把前一 campaign 或 MPS smoke 混入同一研究人口。

## 證據與重現範圍

本輪實際 source 為 commit 5a4319783026e32e4542b8d264b8bc9cb25b55ee；兩階段都是 NVIDIA A100-SXM4-80GB，66 個執行檔案與 frozen 來源相等。固定八組、128 updates、train1024／calibration192／test384 與原生保留人口見 [執行前協定](protocol.md)。HF base/release identities、原始 logits、完整回答、head/features/adapter 與訓練 traces 都分別保存；不把目前 checkout 當成模型執行來源。

原 CPU export 在兩階段都逾時 600 秒，GPU 實驗已完成；只讀 Volume 下載恢復了完整證據：ablate 12,891 檔／57,294,340 bytes，train-recovered 12,900 檔／117,992,663 bytes。原始失敗診斷、transport metadata、每檔校驗及原始嚴格 audit 都保留，資料傳輸成功不會替代品質或校準驗證。

最終本機驗證為 **1,652 passed／3 skipped／1 個既有 warning**，單一 process 93.54 秒；Ruff check、168 檔 format check 與 diff check 全部通過。[逐工具來源與原始測試輸出（ZIP：research/local-validation-final-arithmetic.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip) 綁定測試前後十個 source／test 檔案的相同 SHA。這是本機工具驗證，不代表未完成的雲端算術校驗已通過。

同環境 CPU01 在讀檔校驗階段失敗（4.766 秒），CPU02 在 600 秒上限逾時，完整失敗／source／image／資源與平台紀錄都保留。CPU03 的 [具體輸入包準備（ZIP：research/cpu-arithmetic-03-preflight-v2.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip) 已逐檔核對 20,142 members／84,335,351 bytes，壓縮為 28,341,101 bytes；[自動審核拒絕紀錄（ZIP：research/cpu-arithmetic-03-approval-rejection-full.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip) 顯示上傳與作業均未啟動，需另行明確同意此 payload 至 Modal。剩餘兩組 linkage 保持 inconclusive，不事後放寬原 FP32 門檻。

新 [雲端實驗證據 ZIP](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip) 與 [逐檔 manifest](newcloud-evidence.manifest.json) 收錄完整原始資料及最終報告；其中 reports 保存當時報告，preflight／executed sources／current tools 分別標示。模型 base 的約 24 GB 權重不放入 ZIP；offline 對 HF inventory 做一致性核對，不能聲稱重新讀取全部 base weights。

先前 [授權前 ZIP](raw-evidence.zip) 與 [manifest](raw-evidence.manifest.json) 保持不變：235 members，ZIP SHA256 9358fa7bccf42805eb0f326ee85258801cfa4e96688655e7f3173d6a69bce0e0。它保存執行前 not-run 狀態與本機 MPS 8 cases／29 questions，不包含後來訓練的 head／adapter。MPS paired smoke 不是廣泛品質、校準或速度證明。

2026-10-09 重新核對後，更正 frozen protocol 中「current public300＋sealed1200」的榜單參照文字；原 config／protocol／舊 ZIP 不改寫。Benchmark Heaven [v1.5 method](https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/docs/METHOD-v1.5.md) 是 904 題 open（601 題公開）＋720 題 sealed＝1,624 題，[headline amendment](https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/docs/METHOD-v1.5-ADDENDUM-HEADLINE-A-EQUAL-TYPES.md) 採四軸等權、三型別等權。固定版本的官方文件與 SHA 見 [參照更正（ZIP：research/ranking-reference-20261009/receipt.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip)。本輪 legacy public231 與這套綜合榜不同，sealed 未執行、排行榜未提交，正式 rank／cost／eligibility 仍未知。

另一套 [JevBench coherence 榜](https://jevbench.github.io/) 衡量五個機率一致性維度，mini 為 240 cases／1,200 planned tests／1,248 requests；本輪已使用其固定官方來源，不將兩套榜單或計分方法混用。Formal noninferiority margin 未指定，保持 not_assessed。完整原始機率、source groups、人口支持與失敗項目，才是本輪可核查的研究結論。
