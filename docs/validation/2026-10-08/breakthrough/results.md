# 八組固定優化實驗：完整結果與限制

2026-10-10 僅補正公開下載連結；封存原始版本及當時數值不改寫，後續 CPU 證據見 [10/10 索引](../../2026-10-10/README.md)。

兩階段 GPU 執行與完整資料下載都已完成。此報告使用 [final-eight-profile-audit-v2.json（ZIP：research/final-eight-profile-audit-v2.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip)，六組完整通過結果稽核，head／LoRA 兩組的嚴格機率連結校驗未完成（JSON 狀態為 incomplete_or_not_run）。本輪不採用新 head／LoRA 權重，不改 SDK 預設，也未發布或提交排行榜。

原版 public231 為 197/231；未訓練 USER 提示對照為 200/231，historical Jev-Omni 為 203/231。訓練改善 raw-unit 合成 soft metrics，但各候選獨立校準後落後主要對照，且原生媒體與部分一致性退步。這些收益與代價各自列出，不由公開最好分數選 checkpoint。

沿用 [執行前協定](protocol.md) 與 [固定 config](../../../../configs/experiments/jevbench-breakthrough-20261008.json)：config SHA256 3a8dcceba9fbc0c7ce31bddbb85c0d10acdfa855ddf237f4d5e81982ec6a926d；synthetic manifest SHA256 f408124be6c60ba79983222d1bc64a8432733b7803b69046633fa3d403268d96。實際 source commit 為 5a4319783026e32e4542b8d264b8bc9cb25b55ee，兩階段 66 個執行來源逐 byte／hash 相等。後來的 auditor 與 archive tool 另列 current tools。

[PR #6](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/pull/6) 已於 2026-10-08T05:57:50Z 合併，upstream main 為 3023469aa2579aceeb5f05c496063fac50d20316。contracts 的 Python3.11／3.12／3.13 與 inference CI 成功；外部 reviewer neutral／skipped 不算測試通過。Reconciliation commit 3ed51bb 與 frozen 5a43197 的 tracked tree 都是 229971b08b207240a4fee80cb30241e022ed0c0f；[metadata（ZIP：research/pr6-current.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip) 與 [來源稽核（ZIP：research/project-update-audit.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip) 保留細節。本研究分支沒有新的 push／CI／發布證明。

## Fresh synthetic test384

完整 train1024／calibration192／test384；每型別 calibration64、test128。Soft targets 來自已知有限世界，hard gold 是 oracle 最可能結果，並非真實事件的實現值。Group/id/request fingerprints 分離；test 是六個預先保留模板族，各64個。共享 primitive、labels 與合成生成器，因此只衡量模板遷移，不能聲稱 foundation pretraining 未見或真實世界泛化。外部禁止指紋庫為空，不據此宣稱外部資料零重複。

| Profile | Support | Raw CE / KL / Brier | Calibrated CE / KL / Brier | Score EV-MAE raw / calibrated | T Choice / Noul / Score |
| --- | --- | --- | --- | --- | --- |
| released-current | 384/384；每型別128 | 2.094781 / 1.116134 / 0.379009 | 1.086269 / 0.107622 / 0.065697 | 0.763110 / 0.316624 | 10.000000* / 10.000000* / 5.308844 |
| released-user_question | 384/384；每型別128 | 2.349092 / 1.370444 / 0.419440 | 1.084527 / 0.105880 / 0.068526 | 0.875415 / 0.252551 | 10.000000* / 6.309573 / 10.000000* |
| base-current | 384/384；每型別128 | 2.173162 / 1.194515 / 0.454423 | 1.116496 / 0.137849 / 0.091085 | 1.043469 / 0.386748 | 10.000000* / 10.000000* / 5.956621 |
| base-user_question | 384/384；每型別128 | 2.254476 / 1.275829 / 0.470306 | 1.093758 / 0.115111 / 0.076421 | 0.786512 / 0.262793 | 10.000000* / 10.000000* / 10.000000* |
| released-user-control | 384/384；每型別128 | 2.349092 / 1.370444 / 0.419440 | 1.084527 / 0.105880 / 0.068526 | 0.875415 / 0.252551 | 10.000000* / 6.309573 / 10.000000* |
| released-user-head-init | 384/384；每型別128 | 2.346849 / 1.368202 / 0.419130 | 1.085372 / 0.106725 / 0.069294 | 0.875315 / 0.253125 | 10.000000* / 5.956621 / 10.000000* |
| released-user-head | 384/384；每型別128 | 1.087276 / 0.108628 / 0.067893 | 1.122256 / 0.143609 / 0.086934 | 0.305385 / 0.405810 | 1.678804 / 0.707946 / 0.630957 |
| released-user-lora | 384/384；每型別128 | 1.131608 / 0.152961 / 0.094897 | 1.148564 / 0.169917 / 0.103111 | 0.345642 / 0.377733 | 1.995262 / 1.000000 / 0.841395 |

星號表示固定81點 geomspace(.1,10) 邊界，不擴大搜尋 grid。型別校準只用獨立192題；所有 raw 與 calibrated 的模型 logits 是同一份。數值 grid 核對會保存實際執行 T、最小值 index 與重算 CE，不把本機浮點 representation 取代已執行參數。

| Profile | Type / N | Raw CE / KL / Brier | Calibrated CE / KL / Brier |
| --- | --- | --- | --- |
| released-current | choice / 128 | 2.893285 / 1.655150 / 0.438336 | 1.337156 / 0.099021 / 0.039870 |
| released-current | noul / 128 | 1.428302 / 0.860229 / 0.388809 | 0.687690 / 0.119617 / 0.105193 |
| released-current | score / 128 | 1.962758 / 0.833023 / 0.309881 | 1.233961 / 0.104226 / 0.052028 |
| released-user_question | choice / 128 | 2.764136 / 1.526002 / 0.412555 | 1.332457 / 0.094322 / 0.041072 |
| released-user_question | noul / 128 | 1.376220 / 0.808147 / 0.422539 | 0.712043 / 0.143970 / 0.128925 |
| released-user_question | score / 128 | 2.906918 / 1.777184 / 0.423227 | 1.209081 / 0.079347 / 0.035581 |
| base-current | choice / 128 | 2.135115 / 0.896980 / 0.320002 | 1.349333 / 0.111199 / 0.048347 |
| base-current | noul / 128 | 2.536962 / 1.968889 / 0.683597 | 0.739436 / 0.171363 / 0.155976 |
| base-current | score / 128 | 1.847410 / 0.717676 / 0.359670 | 1.260720 / 0.130985 / 0.068932 |
| base-user_question | choice / 128 | 2.370942 / 1.132808 / 0.410406 | 1.349857 / 0.111723 / 0.055860 |
| base-user_question | noul / 128 | 1.788138 / 1.220065 / 0.606970 | 0.718140 / 0.150067 / 0.135428 |
| base-user_question | score / 128 | 2.604349 / 1.474614 / 0.393542 | 1.213276 / 0.083542 / 0.037975 |
| released-user-control | choice / 128 | 2.764136 / 1.526002 / 0.412555 | 1.332457 / 0.094322 / 0.041072 |
| released-user-control | noul / 128 | 1.376220 / 0.808147 / 0.422539 | 0.712043 / 0.143970 / 0.128925 |
| released-user-control | score / 128 | 2.906918 / 1.777184 / 0.423227 | 1.209081 / 0.079347 / 0.035581 |
| released-user-head-init | choice / 128 | 2.760347 / 1.522213 / 0.412796 | 1.332590 / 0.094456 / 0.041169 |
| released-user-head-init | noul / 128 | 1.371317 / 0.803244 / 0.421351 | 0.714456 / 0.146383 / 0.131162 |
| released-user-head-init | score / 128 | 2.908885 / 1.779150 / 0.423244 | 1.209071 / 0.079336 / 0.035552 |
| released-user-head | choice / 128 | 1.328895 / 0.090760 / 0.038771 | 1.334266 / 0.096131 / 0.042759 |
| released-user-head | noul / 128 | 0.702406 / 0.134333 / 0.119133 | 0.715181 / 0.147108 / 0.129899 |
| released-user-head | score / 128 | 1.230526 / 0.100792 / 0.045776 | 1.317321 / 0.187587 / 0.088145 |
| released-user-lora | choice / 128 | 1.353115 / 0.114980 / 0.047713 | 1.356686 / 0.118552 / 0.052274 |
| released-user-lora | noul / 128 | 0.745526 / 0.177454 / 0.160361 | 0.745526 / 0.177454 / 0.160361 |
| released-user-lora | score / 128 | 1.296184 / 0.166449 / 0.076617 | 1.343479 / 0.213745 / 0.096698 |

六個 heldout 模板的 calibrated 完整分拆如下。完整 raw/calibrated records 與 case/source identities 在原始 archive，不刪除較差的型別或模板。

| Profile | Template / N | Calibrated CE / KL / Brier | Score EV-MAE |
| --- | --- | --- | --- |
| released-current | choice_partition_boxes / 64 | 1.343851 / 0.100629 / 0.038821 | — |
| released-current | choice_route_manifest / 64 | 1.330461 / 0.097414 / 0.040919 | — |
| released-current | noul_access_policy / 64 | 0.691891 / 0.112389 / 0.099816 | — |
| released-current | noul_eligibility_records / 64 | 0.683489 / 0.126845 / 0.110570 | — |
| released-current | score_count_bands / 64 | 1.247685 / 0.108838 / 0.056947 | 0.303946 |
| released-current | score_damage_ledger / 64 | 1.220237 / 0.099614 / 0.047109 | 0.329302 |
| released-user_question | choice_partition_boxes / 64 | 1.342008 / 0.098785 / 0.042475 | — |
| released-user_question | choice_route_manifest / 64 | 1.322905 / 0.089859 / 0.039668 | — |
| released-user_question | noul_access_policy / 64 | 0.704048 / 0.124546 / 0.111141 | — |
| released-user_question | noul_eligibility_records / 64 | 0.720038 / 0.163394 / 0.146709 | — |
| released-user_question | score_count_bands / 64 | 1.216825 / 0.077978 / 0.035140 | 0.220689 |
| released-user_question | score_damage_ledger / 64 | 1.201338 / 0.080715 / 0.036021 | 0.284414 |
| base-current | choice_partition_boxes / 64 | 1.357037 / 0.113814 / 0.048686 | — |
| base-current | choice_route_manifest / 64 | 1.341630 / 0.108583 / 0.048009 | — |
| base-current | noul_access_policy / 64 | 0.702150 / 0.122648 / 0.109703 | — |
| base-current | noul_eligibility_records / 64 | 0.776723 / 0.220079 / 0.202248 | — |
| base-current | score_count_bands / 64 | 1.272341 / 0.133495 / 0.071143 | 0.351831 |
| base-current | score_damage_ledger / 64 | 1.249098 / 0.128475 / 0.066720 | 0.421664 |
| base-user_question | choice_partition_boxes / 64 | 1.363894 / 0.120671 / 0.061985 | — |
| base-user_question | choice_route_manifest / 64 | 1.335820 / 0.102774 / 0.049735 | — |
| base-user_question | noul_access_policy / 64 | 0.693193 / 0.113691 / 0.101077 | — |
| base-user_question | noul_eligibility_records / 64 | 0.743087 / 0.186443 / 0.169779 | — |
| base-user_question | score_count_bands / 64 | 1.221619 / 0.082772 / 0.037519 | 0.230293 |
| base-user_question | score_damage_ledger / 64 | 1.204934 / 0.084311 / 0.038432 | 0.295294 |
| released-user-control | choice_partition_boxes / 64 | 1.342008 / 0.098785 / 0.042475 | — |
| released-user-control | choice_route_manifest / 64 | 1.322905 / 0.089859 / 0.039668 | — |
| released-user-control | noul_access_policy / 64 | 0.704048 / 0.124546 / 0.111141 | — |
| released-user-control | noul_eligibility_records / 64 | 0.720038 / 0.163394 / 0.146709 | — |
| released-user-control | score_count_bands / 64 | 1.216825 / 0.077978 / 0.035140 | 0.220689 |
| released-user-control | score_damage_ledger / 64 | 1.201338 / 0.080715 / 0.036021 | 0.284414 |
| released-user-head-init | choice_partition_boxes / 64 | 1.342246 / 0.099024 / 0.042667 | — |
| released-user-head-init | choice_route_manifest / 64 | 1.322935 / 0.089888 / 0.039672 | — |
| released-user-head-init | noul_access_policy / 64 | 0.706790 / 0.127288 / 0.113630 | — |
| released-user-head-init | noul_eligibility_records / 64 | 0.722121 / 0.165478 / 0.148694 | — |
| released-user-head-init | score_count_bands / 64 | 1.216631 / 0.077785 / 0.035001 | 0.220504 |
| released-user-head-init | score_damage_ledger / 64 | 1.201510 / 0.080888 / 0.036102 | 0.285745 |
| released-user-head | choice_partition_boxes / 64 | 1.343312 / 0.100089 / 0.042941 | — |
| released-user-head | choice_route_manifest / 64 | 1.325220 / 0.092174 / 0.042577 | — |
| released-user-head | noul_access_policy / 64 | 0.721898 / 0.142396 / 0.124357 | — |
| released-user-head | noul_eligibility_records / 64 | 0.708465 / 0.151821 / 0.135441 | — |
| released-user-head | score_count_bands / 64 | 1.350063 / 0.211217 / 0.094905 | 0.386273 |
| released-user-head | score_damage_ledger / 64 | 1.284579 / 0.163956 / 0.081385 | 0.425348 |
| released-user-lora | choice_partition_boxes / 64 | 1.362199 / 0.118977 / 0.050827 | — |
| released-user-lora | choice_route_manifest / 64 | 1.351173 / 0.118126 / 0.053722 | — |
| released-user-lora | noul_access_policy / 64 | 0.712779 / 0.133277 / 0.118999 | — |
| released-user-lora | noul_eligibility_records / 64 | 0.778274 / 0.221630 / 0.201724 | — |
| released-user-lora | score_count_bands / 64 | 1.339918 / 0.201072 / 0.090336 | 0.353170 |
| released-user-lora | score_damage_ledger / 64 | 1.347041 / 0.226418 / 0.103060 | 0.402295 |

另有 [uniform reference（ZIP：research/uniform-reference.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip)：直接對 frozen labels 給等機率，沒有 fitting、模型呼叫或新 checkpoint，不是第九個模型 profile。整體 CE／KL／Brier 為 1.114748／0.136101／0.083021；Score128 EV-MAE 0.389499。兩個訓練候選的 calibrated Brier 都劣於此數學參考，raw-unit 收益仍單獨保留。

## 主要訓練對照與配對區間

Head 主要對照是同52 selected rows／bias 的 FP32 head-init；LoRA 對照是原 BF16 USER control。表中都是 candidate−reference；CE／KL／Brier／EV-MAE 越低越好。完整 source-group bootstrap 固定 seed42、2,000 resamples、95% CI，屬於 descriptive 區間，沒有 multiplicity-adjusted superiority 或事後 noninferiority 門檻。All loss 的 observations/groups=384/384，Score EV-MAE=128/128。

| Comparison | Policy | Type | CE Δ [CI] | KL Δ [CI] | Brier Δ [CI] | Score EV-MAE Δ [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| released-user-head − released-user-head-init | unit | all | -1.259574 [-1.352534, -1.168797] | -1.259574 [-1.352534, -1.168797] | -0.351237 [-0.376880, -0.325894] | -0.569931 [-0.678324, -0.472411] |
| released-user-head − released-user-head-init | unit | choice | -1.431453 [-1.589866, -1.264806] | -1.431453 [-1.589866, -1.264806] | -0.374026 [-0.413376, -0.330864] | — |
| released-user-head − released-user-head-init | unit | noul | -0.668911 [-0.790789, -0.553484] | -0.668911 [-0.790789, -0.553484] | -0.302218 [-0.358365, -0.247462] | — |
| released-user-head − released-user-head-init | unit | score | -1.678359 [-1.820350, -1.546131] | -1.678359 [-1.820350, -1.546131] | -0.377468 [-0.409588, -0.348835] | -0.569931 [-0.678324, -0.472411] |
| released-user-head − released-user-head-init | calibrated | all | +0.036884 [+0.024095, +0.051920] | +0.036884 [+0.024095, +0.051920] | +0.017640 [+0.011602, +0.024040] | +0.152686 [+0.082605, +0.227102] |
| released-user-head − released-user-head-init | calibrated | choice | +0.001676 [-0.001717, +0.005380] | +0.001676 [-0.001717, +0.005380] | +0.001590 [-0.001274, +0.004810] | — |
| released-user-head − released-user-head-init | calibrated | noul | +0.000726 [-0.008711, +0.014071] | +0.000726 [-0.008711, +0.014071] | -0.001263 [-0.009057, +0.009028] | — |
| released-user-head − released-user-head-init | calibrated | score | +0.108250 [+0.074161, +0.144975] | +0.108250 [+0.074161, +0.144975] | +0.052593 [+0.039277, +0.067030] | +0.152686 [+0.082605, +0.227102] |
| released-user-lora − released-user-control | unit | all | -1.217483 [-1.311799, -1.125405] | -1.217483 [-1.311799, -1.125405] | -0.324543 [-0.351031, -0.299728] | -0.529773 [-0.637237, -0.429710] |
| released-user-lora − released-user-control | unit | choice | -1.411022 [-1.570536, -1.243913] | -1.411022 [-1.570536, -1.243913] | -0.364842 [-0.406102, -0.318491] | — |
| released-user-lora − released-user-control | unit | noul | -0.630693 [-0.755659, -0.515448] | -0.630693 [-0.755659, -0.515448] | -0.262177 [-0.318818, -0.205737] | — |
| released-user-lora − released-user-control | unit | score | -1.610735 [-1.746673, -1.480743] | -1.610735 [-1.746673, -1.480743] | -0.346610 [-0.377628, -0.317878] | -0.529773 [-0.637237, -0.429710] |
| released-user-lora − released-user-control | calibrated | all | +0.064037 [+0.052483, +0.076110] | +0.064037 [+0.052483, +0.076110] | +0.034585 [+0.027448, +0.042025] | +0.125181 [+0.074461, +0.182577] |
| released-user-lora − released-user-control | calibrated | choice | +0.024229 [+0.017064, +0.031682] | +0.024229 [+0.017064, +0.031682] | +0.011202 [+0.005897, +0.016831] | — |
| released-user-lora − released-user-control | calibrated | noul | +0.033483 [+0.016690, +0.051230] | +0.033483 [+0.016690, +0.051230] | +0.031436 [+0.015621, +0.048219] | — |
| released-user-lora − released-user-control | calibrated | score | +0.134398 [+0.109148, +0.161450] | +0.134398 [+0.109148, +0.161450] | +0.061117 [+0.049253, +0.073308] | +0.125181 [+0.074461, +0.182577] |

這些結果同時顯示 raw-unit 學習收益與 calibrated 退化。Head train loss 下降不等於部署人口改善；校準192的最佳點也不保證六個 heldout 模板或公開題的最佳機率。不能在已看過的 test384／public231 上另選 T 來合理化結果。

## Legacy public231：公開研究 screening

所有 profiles 都保留完整231 raw responses，attempted/valid=231/231、errors=0；Choice139／Noul74／Score18。使用 [固定官方 public source](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc) scorer 重算。題目與答案本輪前已公開、已觀察，不能用來 fitting、改 recipe 或挑公開最佳 checkpoint。三個政策是 native type-calibrated、unit1、發布 global T=2.7830344470383452 的同 logits replay；CPU replay timing 不代表 GPU latency。

| Profile | Correct calibrated / unit / global | Choice / Noul / Score | Score EV-MAE calibrated / unit / global | False accept / reject | Accuracy Δ vs USER control [95% CI] |
| --- | --- | --- | --- | --- | --- |
| released-current | 197 / 197 / 197 | 120/139；62/74；15/18 | 0.446387 / 0.205581 / 0.225988 | 11/39；1/35 | -1.2987 pp [-4.8458, +2.1645] |
| released-user_question | 200 / 200 / 200 | 119/139；65/74；16/18 | 0.540765 / 0.133807 / 0.130088 | 7/39；2/35 | +0.0000 pp [+0.0000, +0.0000] |
| base-current | 188 / 188 / 188 | 115/139；58/74；15/18 | 0.442017 / 0.218259 / 0.221944 | 5/39；11/35 | -5.1948 pp [-8.8909, -1.6878] |
| base-user_question | 199 / 199 / 199 | 120/139；65/74；14/18 | 0.486136 / 0.178829 / 0.127809 | 5/39；4/35 | -0.4329 pp [-2.9667, +2.1368] |
| released-user-control | 200 / 200 / 200 | 119/139；65/74；16/18 | 0.540765 / 0.133807 / 0.130088 | 7/39；2/35 | reference |
| released-user-head-init | 200 / 200 / 200 | 119/139；65/74；16/18 | 0.540919 / 0.133821 / 0.130299 | 7/39；2/35 | +0.0000 pp [+0.0000, +0.0000] |
| released-user-head | 200 / 200 / 200 | 120/139；66/74；14/18 | 0.183824 / 0.255619 / 0.563580 | 7/39；1/35 | +0.0000 pp [-2.5641, +2.2727] |
| released-user-lora | 197 / 197 / 197 | 117/139；64/74；16/18 | 0.109056 / 0.107438 / 0.279886 | 8/39；2/35 | -1.2987 pp [-3.8470, +1.2501] |

公開配對的 bootstrap 單位是195個 source groups，不把231題當完全獨立。正 temperature 保持這批題目的 argmax；錯誤接受與 Score 機率誤差仍各自保留分母。

| Profile | ECE calibrated / unit / global | Public Brier calibrated / unit / global |
| --- | --- | --- |
| released-current | 0.350988 / 0.130123 / 0.048495 | 0.424774 / 0.274491 / 0.229536 |
| released-user_question | 0.296791 / 0.100116 / 0.046460 | 0.354431 / 0.245086 / 0.210314 |
| base-current | 0.308098 / 0.140283 / 0.060531 | 0.414572 / 0.290358 / 0.235452 |
| base-user_question | 0.291296 / 0.102856 / 0.046579 | 0.334504 / 0.238919 / 0.199781 |
| released-user-control | 0.296791 / 0.100116 / 0.046460 | 0.354431 / 0.245086 / 0.210314 |
| released-user-head-init | 0.292989 / 0.096195 / 0.046455 | 0.353236 / 0.244913 / 0.210263 |
| released-user-head | 0.163199 / 0.104531 / 0.351466 | 0.260195 / 0.215113 / 0.393621 |
| released-user-lora | 0.033203 / 0.074624 / 0.123062 | 0.220504 / 0.225138 / 0.234878 |

Historical Jev-Omni 是前一 campaign 的不同模型／prompt／calibration／硬體公開結果，只有 public231 配對，沒有本輪 fresh384 或 native retention 對照。其203/231不能直接當本輪同 cell control。

| Profile − historical Jev-Omni | Accuracy Δ [95% CI] | Candidate-only / Jev-only correct | N / source groups |
| --- | --- | --- | --- |
| released-current | -2.5974 pp [-6.5797, +1.3217] | 7 / 13 | 231 / 195 |
| released-user_question | -1.2987 pp [-4.9333, +2.5754] | 8 / 11 | 231 / 195 |
| base-current | -6.4935 pp [-10.8696, -2.1643] | 6 / 21 | 231 / 195 |
| base-user_question | -1.7316 pp [-4.8894, +1.3396] | 5 / 9 | 231 / 195 |
| released-user-control | -1.2987 pp [-4.9333, +2.5754] | 8 / 11 | 231 / 195 |
| released-user-head-init | -1.2987 pp [-4.9333, +2.5754] | 8 / 11 | 231 / 195 |
| released-user-head | -1.2987 pp [-4.6812, +2.1459] | 6 / 9 | 231 / 195 |
| released-user-lora | -2.5974 pp [-6.6079, +1.6949] | 9 / 15 | 231 / 195 |

既有 OneJev、Decider、Jev-Omni 與 S1 runtime variants 的完整市場比較另見 [campaign證據](../jevbench/README.md)。不把 registry-only、setup failure 或缺人口的模型當作本輪完成評測。

## 原生與 release retention

每組都完整納入 BoolQ256、native media52（MNIST image32＋FSDD audio20）、historic text128、native fixture8 cases／29 questions。前三項逐題 gold 與答案核對；fixture 是契約 smoke，不能替代完整媒體或廣泛品質。Media／fixture 沒有保存 raw logits，因此不宣稱其 temperature linkage 或 raw replay 已被驗證。

| Profile | BoolQ / 256 | Media / 52 | Historic text / 128 | Fixture / 29 |
| --- | --- | --- | --- | --- |
| released-current | 230 | 33 | 118 | 29/29 valid |
| released-user_question | 231 | 31 | 117 | 29/29 valid |
| base-current | 202 | 33 | 111 | 29/29 valid |
| base-user_question | 220 | 31 | 110 | 29/29 valid |
| released-user-control | 231 | 31 | 117 | 29/29 valid |
| released-user-head-init | 231 | 31 | 117 | 29/29 valid |
| released-user-head | 232 | 27 | 117 | 29/29 valid |
| released-user-lora | 229 | 30 | 117 | 29/29 valid |


| Profile vs USER control | Population | Questions / groups | Accuracy Δ [CI] | Label flips | Max probability drift |
| --- | --- | --- | --- | --- | --- |
| released-current | test | 256 / 256 | -0.390625 [-1.962891, +1.171875] pp | 5 | 0.265522 |
| released-current | media | 52 / 38 | +3.846154 [-6.000000, +14.035773] pp | 22 | 0.145805 |
| released-current | regression_text | 128 / 128 | +0.781250 [-1.562500, +3.906250] pp | 3 | 0.219101 |
| released-current | fixture | 29 / 8 | +0.000000 [+0.000000, +0.000000] pp | 0 | 0.195565 |
| released-user_question | test | 256 / 256 | +0.000000 [+0.000000, +0.000000] pp | 0 | 0.000000 |
| released-user_question | media | 52 / 38 | +0.000000 [+0.000000, +0.000000] pp | 0 | 0.000000 |
| released-user_question | regression_text | 128 / 128 | +0.000000 [+0.000000, +0.000000] pp | 0 | 0.000000 |
| released-user_question | fixture | 29 / 8 | +0.000000 [+0.000000, +0.000000] pp | 0 | 0.000000 |
| base-current | test | 256 / 256 | -11.328125 [-16.015625, -7.031250] pp | 37 | 0.430453 |
| base-current | media | 52 / 38 | +3.846154 [-6.122449, +14.035773] pp | 22 | 0.131212 |
| base-current | regression_text | 128 / 128 | -4.687500 [-11.718750, +2.343750] pp | 20 | 0.424896 |
| base-current | fixture | 29 / 8 | +0.000000 [+0.000000, +0.000000] pp | 0 | 0.228786 |
| base-user_question | test | 256 / 256 | -4.296875 [-8.593750, +0.000000] pp | 33 | 0.409767 |
| base-user_question | media | 52 / 38 | +0.000000 [-7.142857, +7.547170] pp | 11 | 0.054804 |
| base-user_question | regression_text | 128 / 128 | -5.468750 [-10.937500, +0.000000] pp | 13 | 0.461853 |
| base-user_question | fixture | 29 / 8 | +0.000000 [+0.000000, +0.000000] pp | 0 | 0.111477 |
| released-user-head-init | test | 256 / 256 | +0.000000 [+0.000000, +0.000000] pp | 0 | 0.017499 |
| released-user-head-init | media | 52 / 38 | +0.000000 [+0.000000, +0.000000] pp | 3 | 0.002217 |
| released-user-head-init | regression_text | 128 / 128 | +0.000000 [+0.000000, +0.000000] pp | 0 | 0.017458 |
| released-user-head-init | fixture | 29 / 8 | +0.000000 [+0.000000, +0.000000] pp | 0 | 0.013105 |
| released-user-head | test | 256 / 256 | +0.390625 [+0.000000, +1.171875] pp | 1 | 0.411246 |
| released-user-head | media | 52 / 38 | -7.692308 [-19.298246, +3.923529] pp | 26 | 0.512936 |
| released-user-head | regression_text | 128 / 128 | +0.000000 [+0.000000, +0.000000] pp | 0 | 0.329838 |
| released-user-head | fixture | 29 / 8 | +0.000000 [+0.000000, +0.000000] pp | 0 | 0.319695 |
| released-user-lora | test | 256 / 256 | -0.781250 [-1.953125, +0.000000] pp | 2 | 0.326842 |
| released-user-lora | media | 52 / 38 | -1.923077 [-5.882353, +0.000000] pp | 7 | 0.591307 |
| released-user-lora | regression_text | 128 / 128 | +0.000000 [+0.000000, +0.000000] pp | 0 | 0.251613 |
| released-user-lora | fixture | 29 / 8 | +0.000000 [+0.000000, +0.000000] pp | 0 | 0.327953 |

所有 population 完整配對，不縮成人口交集作成功結論。Accuracy 相同也不等於 distributions 或 labels 全相同；完整 flips／drift 都保留。Formal noninferiority margin 未指定，保持 not_assessed。

[完整訓練覆蓋與media分拆（ZIP：research/final-coverage-diagnostic.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip)：head-init→head 31→27，7個新錯／3個修正；gold0–7的42題27→27（3新錯／3修正），gold8／9的10題4→0（4新錯）。Image28→25，audio3→2。LoRA31→30只有1個新增錯誤，屬gold0–7音訊；gold8／9仍4→4。Head bias只有positions0–7非零，但初始W未獨立保存，不據此宣稱W parity或因果。覆蓋缺口不能解釋所有退步，也不能正當化事後fallback。

## Strict coherence 與溫度

每個 profile／policy 都有240 cases、50 checks、1,200 planned tests、1,248 cached answers；原始 outcome rows=1,468、errors=0。一次 request 可有多問題，本輪兩個訓練 profile 的逐 request numeric diagnostic各 policy 為12,072個 questions／33,071個機率。使用 [固定官方 coherence source](https://github.com/JevBench/jevbench/tree/e18733694623aa93058e279c3534f8b8e2edefa3) 完整 raw-answer replay，不以 graded 或某一維代替 strict overall。

| Profile | Overall cal / unit | BAT cal / unit | MEA cal / unit | REP cal / unit | LOG cal / unit | CHO cal / unit |
| --- | --- | --- | --- | --- | --- | --- |
| released-current | 71.91% / 81.86% | 100.00% / 100.00% | 23.04% / 61.52% | 73.61% / 83.33% | 87.92% / 90.83% | 75.00% / 73.61% |
| released-user_question | 65.87% / 81.78% | 100.00% / 100.00% | 15.44% / 61.52% | 73.06% / 86.11% | 74.17% / 90.42% | 66.67% / 70.83% |
| base-current | 65.79% / 75.23% | 100.00% / 100.00% | 18.14% / 50.74% | 65.83% / 75.83% | 82.50% / 82.92% | 62.50% / 66.67% |
| base-user_question | 66.90% / 82.77% | 100.00% / 100.00% | 20.59% / 63.73% | 75.56% / 86.67% | 75.83% / 91.25% | 62.50% / 72.22% |
| released-user-control | 65.87% / 81.78% | 100.00% / 100.00% | 15.44% / 61.52% | 73.06% / 86.11% | 74.17% / 90.42% | 66.67% / 70.83% |
| released-user-head-init | 65.84% / 81.87% | 100.00% / 100.00% | 15.44% / 62.01% | 72.50% / 86.11% | 74.58% / 90.42% | 66.67% / 70.83% |
| released-user-head | 56.13% / 60.58% | 100.00% / 100.00% | 17.89% / 24.02% | 62.50% / 64.17% | 68.33% / 67.50% | 31.94% / 47.22% |
| released-user-lora | 70.79% / 75.08% | 100.00% / 100.00% | 40.20% / 50.00% | 73.06% / 77.78% | 83.75% / 83.75% | 56.94% / 63.89% |


| Profile | Temperature linkage status | Normalized entropy cal / unit | Max-confidence cal / unit |
| --- | --- | --- | --- |
| released-current | verified_same_raw_logits | 0.917375 / 0.073673 | 0.577950 / 0.973190 |
| released-user_question | verified_same_raw_logits | 0.787182 / 0.052716 | 0.667340 / 0.978709 |
| base-current | verified_same_raw_logits | 0.913266 / 0.135426 | 0.575495 / 0.952633 |
| base-user_question | verified_same_raw_logits | 0.845198 / 0.052582 | 0.640418 / 0.977901 |
| released-user-control | verified_same_raw_logits | 0.787182 / 0.052716 | 0.667340 / 0.978709 |
| released-user-head-init | verified_same_raw_logits | 0.771882 / 0.052686 | 0.674521 / 0.978671 |
| released-user-head | integrity_failed | 0.564755 / 0.615740 | 0.781641 / 0.770252 |
| released-user-lora | integrity_failed | 0.220103 / 0.172376 | 0.916652 / 0.936725 |

凍結來源設計為同 logits 的 calibrated／unit 兩個政策；其中 head／LoRA 的嚴格機率連結校驗仍未通過，不能把其溫度效果視為完整驗證。LoRA calibrated overall 比 USER control 高約4.92pp，但 unit overall 低約6.69pp；兩者同時呈現，不能把 weight＋temperature 的複合差異當純訓練進步。高 T 有推近 uniform 的效應，而 calibration不是coherence，Score EV 誤差與LOG強項需一併保留。

## 執行、權重與可重現證據

| Stage | GPU / call | Original CPU export | Read-only recovered transport |
| --- | --- | --- | --- |
| ablate | A100-SXM4-80GB / fc-01M4DTT0FTMVBPE6HDT23DV6J4 | 600.001502秒 timeout；原CLI exit1 | 12,891 files／57,294,340 bytes／66 sources |
| train | A100-SXM4-80GB / fc-01M4DZVD9R2ER2KKRBQHEF8DME | 600.001191秒 timeout；原CLI exit1 | 12,900 files／117,992,663 bytes／66 sources |

Train GPU call successful，elapsed6008.922731秒；CPU export 失敗與 GPU 訓練成功分別記錄。train-recovered 是 logical train 的完整只讀恢復目錄，不覆寫原 canonical train；原 timeout、兩次 transfers、pending/final官方平台診斷皆在 archive。Recovery開始時原export仍執行中，所以其 original_export_failure=not_supplied，最終 timeout另存診斷，不事後改寫 transport metadata。

Head 最終52×3840 FP32 weights／52 bias與1024×3840 FP32 detached features已檢查有限值、來源順序、groups、labels、soft targets、Score values與SHA。Head128 finite loss updates；原helper沒有逐步 gradient norms或 sampled case IDs，保持 unrecorded，不憑 seed補成已觀測紀錄。LoRA128 optimizer updates／accumulation8＝1024 unique exposures，seed42完整shuffle順序、finite losses／preclip gradient norms、recipe與10,665,984個FP32 attention adapter參數皆逐項核對。對 unused media branches 不宣稱已觀察 gradients 或完成原生訓練。

LoRA保存 final128 adapter（42,718,528 bytes）、config、README與完整updates/training；每32步覆寫同一adapter目錄，因此沒有全部中間 binary checkpoints。本輪只評宣告的final checkpoint。HF release為knowlet/Gemma-4-12B-Unified-System-One@a66f836b56605039fe040f330180e336d19b3362，base為google/gemma-4-12B-it@707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7。Offline確認其inventory/processor identities自洽，未重新讀取約24GB base權重；不將metadata核對說成完整獨立權重重hash。

第一次 [八組嚴格 audit（ZIP：research/final-eight-profile-audit.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip) 保持不變，當時六組complete／兩組未通過：head校準點跨平台差一個binary64 ULP；head/LoRA各5/4個 coherence questions超過原1FP32-epsilon門檻，最大差異1.7881393433e-7。完整 [numeric diagnostic（ZIP：research/final-audit-numeric-diagnostic.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip) 保存全部差異、raw keys與alternate references；不選3ULP作事後放寬。頭部grid修正僅接受同index的相鄰interior representation，端點維持exact、CE abs1e-12/rel0，使用actual executed T重算。原macOS/ARM Torch2.8與雲端Linux/x86 Torch2.10校驗範圍需分別讀最終audit與CPU proof，不宣稱逐bit parity。

本輪沒有完成同環境CPU arithmetic proof。CPU01在讀檔校驗階段失敗（4.766秒），CPU02大量遠端讀檔在600秒上限逾時；兩次failure／source／image／resources／logs皆保存。新CPU03已準備同20,142members／84,335,351bytes的SHA-bound快照，壓縮28,341,101bytes，source2374721833911e7349f985928b070b3de937113ee418f79208bf6fe5196741c1，但上傳與作業在啟動前被自動審核拒絕，需要human consent，保持not_run。沒有藉此放寬原1FP32-epsilon門檻；head／LoRA兩個coherence temperature linkages保持inconclusive，不能聲稱完整8/8outcome audit或Mac嚴格parity。

CPU03 [準備與來源綁定（ZIP：research/cpu-arithmetic-03-preflight-v2.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip) 可檢查原sameModalVolume的已完成transferreceipt、每檔size／SHA、原GPUterminalreceipt及唯一inputZIP；這是待批准的read-only CPU重播，未增加GPU訓練、HF下載、model forward或Volume write。

[完整本機驗證（ZIP：research/local-validation-final-arithmetic.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip)：單一process 1,652 passed／3 skipped／1 existing warning，93.54秒、exit0；source commit 6c3c4750fc428c034e8ae479989c75b7dafc783b。Skips為SentenceTransformer1個與bitsandbytes2個，warning為既有Starlette TestClient deprecation。Ruff check／format與diff check狀態同時保存。這是local tool/runtime測試，與GPU profiles品質結論分開。

完整原始證據與當時reports收錄於 [newcloud-evidence.zip](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip) 和 [逐檔SHA/size/CRC manifest](newcloud-evidence.manifest.json)，含兩階段raw/cal/public/coherence全人口、head/features/adapter、traces、sources、recoveries、失敗與最終稽核。舊授權前 [ZIP](raw-evidence.zip)／[manifest](raw-evidence.manifest.json) 不覆寫。

## 下一輪可驗證的優化方向

1. 補齊原生影像／音訊與十一候選位置的獨立訓練資料，candidate order做受控重排，保留真實labels映射。現有train/cal/test全是文字，positions8/9/10沒有direct loss exposure；runtime支援11-way，這是訓練覆蓋缺口。新media需與52題保留集依原始來源／speaker／圖片ID分割，也不使用公開或coherence測試來訓練。
2. 使用Noul／Choice／Score與原版保留資料的混合訓練，先固定對照來檢查收益與忘記原生能力的代價。歷史release只用2048 BoolQ Noul cases；Jev-Omni [固定model card](https://huggingface.co/akhilaaa3/Jev-Omni/blob/5addda86ddee081a68fb067477ea100c221b8917/README.md)聲明30,000問題微調，該聲明未被獨立訓練稽核，資料量差異不是因果證明。
3. 新建貼近部署modalities／candidate cardinalities／Score levels的獨立校準人口；事先固定unit／global／type政策及proper losses對照，檢查校準192到heldout模板、公眾題與媒體的遷移。保留raw-unit訓練收益，避免只靠高T推近uniform或以公開成績選T。
4. 保留BoolQ256、media52、historic128、fixture8/29、fullpublic與五個coherence維度；新recipe先固定margin、checkpoint與完整支持條件，再進行新的雲端訓練。本輪 frozen protocol與八組final checkpoint保持不變。

這些是後續研究假說，未當成本輪已執行的額外GPU profile或已驗證改善。Formal noninferiority=not_assessed；sealed=not_run、submission=not_submitted，正式rank／cost／eligibility未知。CPU replay、不同A100 cell或MPS fixture smoke不能證明GPU speedup或榜單資格。

## 榜單參照更正（2026-10-09）

Frozen protocol 中 current public300＋sealed1200 的參照文字不符合本次重新核對的官方方法；原config／protocol／舊archive保持不變，這是對外部榜單參照的更正，不是事後更改實驗人口或scorer。Benchmark Heaven [v1.5方法](https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/docs/METHOD-v1.5.md) 為904題open（601公開）＋720題sealed＝1,624題；[headline amendment](https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/docs/METHOD-v1.5-ADDENDUM-HEADLINE-A-EQUAL-TYPES.md) 改為四軸各25%、三型別各1/3。固定primary-source版本／SHA與對照見 [receipt（ZIP：research/ranking-reference-20261009/receipt.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip)。本輪沒有完整v1.5人口、正式速度或定價證據，不能推算該綜合榜名次。

另一套 [JevBench coherence榜](https://jevbench.github.io/) 比較240cases／1,200tests的五維一致性，並另報準確率；高coherence也可能來自uniform輸出，所以不能據本輪unit81.86%的點估計宣稱第二名或保證上榜。本輪八組已完成固定官方 mini 的 raw-answer 評分重算，但兩組 raw-logit 機率連結仍未完成；尚無榜單接納或提交證明。這兩套同名benchmark明確分開。
