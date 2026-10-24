# GX10 Decision3 CPU16 真實小測試

16核心／4GiB的新設定完成一次真正模型載入、兩次四列BF16 CPU推論及原始資源清理；主程序封閉審查通過，範圍限於此小測試。短輸入沒有觀測到加速。

| 同一四列135-token小測試 | 先前CPU4 | 新CPU16 |
| --- | ---: | ---: |
| 第一次完整推論（含耐久紀錄／timer cleanup） | 0.917990秒 | 1.204805秒 |
| 第二次完整推論 | 0.900895秒 | 1.094404秒 |
| 外部啟動到結束 | 16.146155秒 | 17.296471秒 |

CPU4使用16–19，CPU16使用4–19。這是各兩次的觀測，未完成配對重複實驗，不能把差異全部歸因於核心數。Linux核心capacity／最高頻率／cluster快照顯示，CPU16混合較低capacity核心並跨兩個cluster；舊CPU4均在同一個較高capacity群組。同步、核心配置與跨群組成本是待驗證假說。接下來會另外驗證同一群15–19的五顆快核心與代表性人工輸入長度，據實選型。

新小測試的原模型版本、BF16 backbone、FP32原head、temperature1、非causal batch4數學、context16384拒絕不截斷、兩次呼叫與120／20秒界限均保留。實際shape為[4,135,1024]，沒有修改tensor值、選項、機率或答案。啟動門檻為R+完整4GiB，swap0，CPU quota1600%，intra-op16／inter-op1；host R維持26119172916 bytes。觀測到原unit peak1061474304 bytes，原事件計數無OOM；這個小測試不證明長context或模型加Chromium的容量。

owner08的48項假資料回歸與native04的24項檢查、Ruff／AST／預設dry／plan builder均通過。controller相對owner07的精確16項資源替換通過，formatter未改controller bytes。原producer freeze與pending authoring紀錄保留原狀；當前狀態由新的validation、執行與[封閉審查](../../../artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-04/root-closed-smoke-result-01.json)說明。

主程序核對原root0／owner.completed=true、current nonce／PID-start／admitted cohort／command argv與55個選定來源、20個model與28個runtime紀錄、兩次完整機率／argmax／entropy confidence、原始形狀與期限，以及原始正向authenticated_stop、兩次空掃描與cgroup消失。runtime保留原SHA及stat-only範圍，不宣稱完整native DSO封閉。沒有第二位審查者直接讀原件；獨立審查限於來源，四項來源問題修正後由主程序驗證實際資料。

這輪沒有benchmark或browser分數、沒有訓練或競品勝利，也尚未選作full07的前提。精確值、版本與原始證據SHA見[JSON](decision3-cpu16-smoke-results.json)。原始請求／答案／機率／權重／特徵不公開。
