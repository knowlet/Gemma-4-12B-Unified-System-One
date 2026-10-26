# GX10 Decision3 五核心真實短測

同一群組的五核心設定完成真正模型載入與兩次四列推論，封閉審查通過。

| 同一四列短輸入 | 先前四核心 | 先前十六核心 | 新五核心 |
| --- | ---: | ---: | ---: |
| 第一次完整呼叫 | 0.917990 秒 | 1.204805 秒 | 0.843083 秒 |
| 第二次完整呼叫 | 0.900895 秒 | 1.094404 秒 | 0.807527 秒 |

新設定使用 CPU15–19、quota500%、intra-op5／inter-op1、4GiB／swap0，模型仍為固定版本的 d3-edge，BF16 backbone、原 FP32 head、temperature1、batch4、context16384 及原120／20秒限制均保留。實際第一層形狀為[4,135,1024]，載入一次；外部啟動到結束16.258554秒，原unit記憶體峰值1022169088 bytes，事件計數無OOM。

這次兩筆觀測比先前四核心與十六核心短測快，但尚未完成配對重複實驗，不能推論長輸入、整套評測或瀏覽器都會加速。模型加Chromium共用4GiB的容量也尚未驗證。

新owner的四項資源檢查、精確父版差異，以及原生程式的30項假資料檢查、Ruff、AST、預設dry與只讀metadata的plan builder通過。主程序核對原root與owner成功、目前nonce與child身份、55個來源／20個model／28個runtime紀錄、完整原生機率與形狀、期限及原始authenticated_stop正向清理。第二位審查者僅審查來源，未直接讀這次原件；runtime保留原SHA及stat-only範圍。

這輪沒有新增兩套JevBench分數或瀏覽器分數，也沒有訓練或競品勝利。下一步是將這個設定接上真正HTTP與網頁點擊。精確數值及證據SHA見[JSON](decision3-cpu5-smoke-results.json)。原始狀態、回應、權重與特徵不公開。
