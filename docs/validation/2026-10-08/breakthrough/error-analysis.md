# S1 independent 與 Jev-Omni：完整人口錯誤診斷

這是已曝露公開資料的開發診斷，沒有訓練、溫度 fitting、checkpoint 選擇或重新推論。保留原六個 primary profiles；本次聚焦 run02 `s1-independent` 與 run03 `jev-omni`。兩者 GPU 標籤均為 NVIDIA A100 80GB PCIe，仍是不同 cell、模型／prompt／head 配置，不能據此推斷訓練來源或速度因果。

完整 231 題中，S1 為 197/231，Jev-Omni 為 203/231。觀察差為 2.5974 個百分點，231 paired cases／195 source groups 的描述性 95% CI 是 [-1.2931, +6.7228] 個百分點，跨過零。分類結果不等於已證明真實能力排名。

不是 17 題差異：190 題兩者都對，21 題兩者都錯，13 題只有 Jev 對、7 題只有 S1 對；20 題 correctness discordants 涵蓋 19 source groups，淨差 6 題。總 predicted-label disagreements 是 25，其中 5 題兩者都錯但錯誤 label 不同。

六個 primary public runs 的每份 233 個 declared artifacts、231 原始 requests／answers／官方 outcomes／summary 都重新核對；五份 coherence 原答案 cache 各 1,248 個 hash 與 canonical audit 相符。五份 saved raw outcomes 另用 pinned 官方 judge 重算，都是 240 cases、50 checks、1,200 grouped tests、1,468 raw outcomes、0 errors。Scorer 的 fixed tolerance 仍為 0.05。Compiled coherence 保留 expected_not_run。

## 全部 231 題的拆分

**型別**

| 分類 | N | S1 正確 | Jev 正確 | Jev−S1 題數 |
| --- | --- | --- | --- | --- |
| choice | 139 | 120/139 | 123/139 | +3 |
| noul | 74 | 62/74 | 66/74 | +4 |
| score | 18 | 15/18 | 14/18 | -1 |

**Tier／來源 JSONL**

| 分類 | N | S1 正確 | Jev 正確 | Jev−S1 題數 |
| --- | --- | --- | --- | --- |
| easy | 48 | 48/48 | 48/48 | +0 |
| hard | 111 | 80/111 | 85/111 | +5 |
| original | 72 | 69/72 | 70/72 | +1 |

**Category（原 family）**

| 分類 | N | S1 正確 | Jev 正確 | Jev−S1 題數 |
| --- | --- | --- | --- | --- |
| adequacy | 12 | 9/12 | 11/12 | +2 |
| adversarial | 6 | 6/6 | 6/6 | +0 |
| ambiguous | 7 | 5/7 | 5/7 | +0 |
| extraction | 24 | 24/24 | 24/24 | +0 |
| fact | 12 | 12/12 | 12/12 | +0 |
| intent | 24 | 24/24 | 24/24 | +0 |
| judge_hard | 17 | 12/17 | 14/17 | +2 |
| long_policy | 19 | 12/19 | 15/19 | +3 |
| multi_hop | 18 | 14/18 | 15/18 | +1 |
| ordinal | 12 | 12/12 | 12/12 | +0 |
| policy | 12 | 12/12 | 11/12 | -1 |
| probability | 10 | 7/10 | 8/10 | +1 |
| routing | 12 | 12/12 | 12/12 | +0 |
| routing_hard | 5 | 5/5 | 5/5 | +0 |
| temporal_numeric | 15 | 6/15 | 4/15 | -2 |
| tool_selection | 12 | 12/12 | 12/12 | +0 |
| tradeoff | 6 | 5/6 | 5/6 | +0 |
| trap | 8 | 8/8 | 8/8 | +0 |

**選項／levels 數量**

| 分類 | N | S1 正確 | Jev 正確 | Jev−S1 題數 |
| --- | --- | --- | --- | --- |
| 2 | 74 | 62/74 | 66/74 | +4 |
| 3 | 15 | 9/15 | 10/15 | +1 |
| 4 | 70 | 63/70 | 63/70 | +0 |
| 5 | 56 | 50/56 | 51/56 | +1 |
| 6 | 16 | 13/16 | 13/16 | +0 |

**State 格式**

| 分類 | N | S1 正確 | Jev 正確 | Jev−S1 題數 |
| --- | --- | --- | --- | --- |
| plain_text | 196 | 173/196 | 177/196 | +4 |
| structured_json | 35 | 24/35 | 26/35 | +2 |

Noul +4、Choice +3、Score −1 組成了淨差 +6。差距主要出現在 hard tier（80/111 對 85/111），不是 easy tier（兩者 48/48）。long_policy +3、adequacy +2、judge_hard +2 是 Jev 的淨優勢；temporal_numeric 是 S1 +2。兩者共同錯的 21 題包含 temporal_numeric 8 題、long_policy 3 題、judge_hard 3 題，表示可改善空間不限於那 6 題淨差。

Noul gold="no" 共 39 題：S1 錯誤接受 11/39，Jev 6/39；gold="yes" 共 35 題：S1 錯誤拒絕 1/35，Jev 2/35。這是全體 74 題的對稱檢查，不是只挑選 Jev 勝出的反例。

長度分組採原文字數；structured JSON 使用排序鍵的 UTF-8 JSON 字串字數，並非模型 tokenizer 的真實 token 數。8,000+ characters 的 36 題，S1 25/36、Jev 29/36；2,000–7,999 的 13 題 S1 7/13、Jev 6/13，因此不能宣稱字數越多就必然造成差距。JSON 內另外保留所有 source、作者 metadata、type×option_count 與長度分組。作者欄只是資料集 provenance：claude-opus-5 54 題、gpt-5.6-sol 57 題、未聲明作者 120 題；它不是任何被測模型的訓練來源證據。

## 機率、信心與 Score 誤差

| 模型 | Accuracy | 官方 Brier | 官方 ECE | 平均 top confidence | 平均 gold mass | confidence≥.9 的錯題/N |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | 85.28% | 0.229536 | 0.048495 | 0.8388 | 0.7578 | 3/113 |
| Jev | 87.88% | 0.165866 | 0.049663 | 0.9181 | 0.8580 | 6/179 |

Jev 分布較尖、gold mass 較高且 Brier 較低；S1 的整體 ECE 稍低。兩者高信心範圍覆蓋不同人口，不能拿 3/113 與 6/179 直接宣稱固定路由策略效果。JSON 保存全部 231 題的 top probability、gold probability、top-two margin、原始分布與四個固定 confidence bins。

Score 只有 18 題，S1 15/18、Jev 14/18；官方 expected-value MAE 分別 0.225988 與 0.241852。Signed EV error 平均為 +0.138031／+0.239874，是額外診斷而非官方 metric。正確性依 argmax level 計，不是把 EV 四捨五入。

S1 有一題 exact top tie：`hard-opus-a-temporal_numeric-07` 的 sep_26/sep_27 同為最大機率，官方 lexical rule 選 gold sep_26，所以仍依原規則計為 S1-only correct。Jev 也有一題 exact tie（`hard-sol-b-temporal_numeric-04`），但兩個 tied labels 都不是 gold，屬共同錯題。這提醒我們低 margin 的題目要檢查完整分布，不能只看單一 argmax。

正的 scalar temperature 若對同題所有 logits 等比例縮放，不改 argmax；它可以改信心卻無法單獨改正這些 classification 錯誤。本報告不能作 fitting T 的資料。

## 所有 20 個 correctness discordants

下表是完整列表；信心欄是各自 predicted label 的 top probability，非 gold probability。JSON 同時保存 gold probability、完整 native probabilities、raw path／SHA 和 authored rationale。

| Task ID | 只有誰正確 | Type | Category | Options | Gold | S1 predicted / confidence | Jev predicted / confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original-policy-06-1 | S1 | noul | policy | 2 | yes | yes (0.8149) | no (0.5756) |
| original-adequacy-03-0 | Jev | noul | adequacy | 2 | no | yes (0.7287) | no (0.8912) |
| original-adequacy-03-1 | Jev | noul | adequacy | 2 | no | yes (0.8783) | no (0.9343) |
| hard-opus-a-long_policy-04 | Jev | choice | long_policy | 5 | cfo | vp_and_finance_director (0.7382) | cfo (0.6923) |
| hard-opus-a-long_policy-17 | Jev | choice | long_policy | 5 | database_oncall | no_page_ticket_only (0.6349) | database_oncall (0.9819) |
| hard-opus-a-long_policy-19 | Jev | noul | long_policy | 2 | no | yes (0.7545) | no (0.7520) |
| hard-opus-a-probability-07 | Jev | choice | probability | 3 | escalated | resolved_first_contact (0.8684) | escalated (0.9587) |
| hard-opus-a-probability-08 | S1 | noul | probability | 2 | no | no (0.5889) | yes (0.6192) |
| hard-opus-a-temporal_numeric-07 | S1 | choice | temporal_numeric | 5 | sep_26 | sep_26 (0.3141) | sep_28 (0.2902) |
| hard-opus-a-temporal_numeric-09 | Jev | noul | temporal_numeric | 2 | no | yes (0.8216) | no (0.7202) |
| hard-opus-b-probability-04 | Jev | choice | probability | 3 | attended | no_show (0.5327) | attended (0.8016) |
| hard-opus-c-long_policy-05 | S1 | score | long_policy | 4 | 1 | 1 (0.4228) | 3 (0.5756) |
| hard-opus-c-temporal_numeric-12 | S1 | choice | temporal_numeric | 3 | expired_30_month_cap | expired_30_month_cap (0.5516) | expired_24_month_term (0.6598) |
| hard-sol-a-multi_hop-05 | S1 | choice | multi_hop | 4 | require_director | require_director (0.6082) | pay_manager_approved (0.7901) |
| hard-sol-a-multi_hop-10 | Jev | choice | multi_hop | 4 | approve_45_days | reduce_to_30_days (0.6923) | approve_45_days (0.8536) |
| hard-sol-a-multi_hop-12 | Jev | choice | multi_hop | 4 | needs_authentication | covered_recurrence (0.6244) | needs_authentication (0.7636) |
| hard-sol-b-long_policy-06 | Jev | choice | long_policy | 4 | east_ward | no_award (0.4921) | east_ward (0.9072) |
| hard-sol-b-temporal_numeric-01 | S1 | choice | temporal_numeric | 4 | late_by_under_2h | late_by_under_2h (0.3108) | within_window (0.4902) |
| hard-sol-c-judge_hard-07 | Jev | noul | judge_hard | 2 | no | yes (0.5997) | no (0.5603) |
| hard-sol-c-judge_hard-13 | Jev | noul | judge_hard | 2 | no | yes (0.7627) | no (0.7621) |

原始 adequacy 的兩個 Jev-only 案例是同一 source group 的不同表述：要求 JSON array，回應卻是 object。其餘 Jev-only 案例的 frozen rationale涉及前提缺漏／生效來源與時間、條件人口篩選、定義／排除條款、單位或 rounding 的局部錯誤。這些只是資料中的可核對失誤，不是由結果反推訓練原因。S1-only 案例同樣保留：截止點的 inclusive boundary、條件救援機率、淨額 threshold、ISO week／時區／上限、長文 licence tally、另一個前提 footnote。

## Coherence 五個 dimensions 與型別支持

| Dimension | 意義 | Grouped tests | S1 causal | S1 independent | Jev |
| --- | --- | --- | --- | --- | --- |
| REP | 表示一致性 | 360 | 58.33% | 73.06% | 81.39% |
| BAT | 批次獨立性 | 120 | 38.33% | 96.67% | 100.00% |
| MEA | 機率測度一致性 | 408 | 15.44% | 37.99% | 49.26% |
| LOG | 邏輯一致性 | 240 | 69.58% | 85.42% | 75.42% |
| CHO | 選項集合一致性 | 72 | 20.83% | 66.67% | 66.67% |

Independent 的 BAT 是 116/120 tests 通過；batch_order 24/24、其 222 raw outcomes 的最大差值全部為 0。但 MEA 只有 155/408 tests 通過；REP 的 original Score targets 為 8/21 通過，原 Choice targets 122/165；這是 type 支持的診斷，不是重新定義官方 dimension score。官方 overall 是五個 dimension 等權平均，不能拿所有 1,468 raw outcomes 的 pooled 通過率替代。

MEA 的 original-target 支持：Choice 192 tests／142 cases，S1 88/192、Jev 116/192；Noul 120 tests／101 cases，S1 35/120、Jev 50/120；Score 96 tests／88 cases，S1 32/96、Jev 35/96。Original target type 是套件原始題型，變換後題型可能不同；病例與 tests 相關，並非獨立重複實驗。

具體 checks（每項完整 24 tests、24 cases，固定官方 tolerance）：

| Relation | 一般語意 | S1 passes /24 | Jev passes /24 | S1 mean max deviation | Jev mean max deviation |
| --- | --- | --- | --- | --- | --- |
| description_swap | descriptions exchanged, ids fixed | 6 | 15 | 0.2587 | 0.1185 |
| noul_as_choice | Noul versus yes/no Choice | 16 | 24 | 0.0440 | 0.0018 |
| noul_as_score | Noul versus two-level Score | 16 | 24 | 0.0916 | 0.0022 |
| choice_complement | Noul 'not o' = 1 - P(o) | 0 | 1 | 0.6624 | 0.7580 |
| score_cumulative | 'level k or higher' = upper tail, every k | 0 | 3 | 0.3302 | 0.3048 |
| choice_indicator | Noul 'is it o?' = P(o), every o | 1 | 4 | 0.4144 | 0.4019 |
| compound_negation | P(not (A and B)) + P(A and B) = 1 | 0 | 0 | 0.7064 | 0.4110 |
| natural_negation | P(Q) + P(Q') = 1, natural opposite | 11 | 19 | 0.1329 | 0.0555 |
| batch_order | question order reversed | 24 | 24 | 0.0000 | 0.0000 |

Complement／marginalisation 失敗涵蓋 Noul、Choice、Score，而且 deviation 常遠大於 0.05；它們不是只靠浮點 epsilon 即可解釋。Jev 在這些 measure checks 也很弱，並未提供可直接複製的完美解法。S1 的 LOG 85.42% 高於 Jev 75.42%，Score accuracy／EV-MAE 也較佳；任何改善都應守住這些強項。

## 三個可測的方向

1. **通用 typed probability consistency。** 在獨立建構的新資料與凍結、不重疊的測試人口上，加入同一事件的 Noul／binary Choice／two-level Score、否定、partition 與 cumulative views 的一致性目標。訓練一般機率與語義關係，不辨認 benchmark relation 名称、不對公開案例或選T。先看是否能改善 MEA／REP.type，同時維持 Noul／Choice／Score accuracy、Brier、EV-MAE 與 LOG。

2. **前提完整性與 criterion binding。** 新 curriculum 涵蓋 missing prerequisites、格式／完整性錯誤、來源時效／權限、inclusive boundary、長文條款；用中性 option ids 與 shuffled descriptions 讓模型學會按 criteria／證據決策。39 題 negative Noul 的 11 false acceptances、long_policy 的 12/19、description_swap 的 6/24 是可驗證的需求，不構成因果訓練來源證據。必須在新、獨立的 typed／multimodal 人口上驗證，而非只修那 13 題。

3. **候選 head 投影精度的有限消融。** Executed source 相符的 `src/s1/engine.py:78` 與 `src/s1/unified.py:51–57` 都先以 head weight dtype 做 candidate linear，再 cast logits 到 float32。以同 checkpoint／prompt／既有 temperature，在同一 A100 cell 做 selected candidate-row FP32 投影對照，量測全人口機率／labels／ties／校準与 latency。只有一個 S1 top tie，不能推斷 6 題差距由 BF16 造成；這只是低成本可否證假說。Jev 的 FP32 head 也是不同 architecture／prompt 的配置，沒有因果證明。

本報告沒有變更模型、app、code、資料或舊報告。完整資料與輸入 hashes 在新 JSON；canonical audit 仍是先前執行證據。後續接受優化必須另用新的 held-out／calibration split 與適當完整人口 gates，不能把公開診斷當作新的公信力成績。

Input inventory SHA256: `9a1ac241e9a282877adac77f6932fc50a4e9a04673c4cc840ad428b27be1f830`。Public source pin: `bb05a335bc809e61b20c0f745d25499a82b326fc`；coherence source pin: `e18733694623aa93058e279c3534f8b8e2edefa3`。
