# JevBench 評測證據索引

更新日期：2026-10-08（Asia/Taipei）。**封存狀態：run01／02／03 已完成下載；完整證據已封存並逐檔驗證。**
此目錄記錄公開評測、coherence 評測及推論實驗；沒有寄送資料或提交排行榜。
正式排名、sealed evaluation 與成本都仍未取得，不能把缺值解讀為零。

## 評測範圍與來源

- `legacy_public231` 保留三份公開檔案全部 231 題及原順序，使用發布者的 scoring。
  [來源固定在 bb05a335](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc)，
  資料與程式雜湊見 [jevbench-public.json](../../../../configs/benchmarks/jevbench-public.json)。
  公開資料可能已被模型訓練或先前調整使用，結果只代表公開 screening。
- `jevbench-mini` 評估 coherence 與 stability。它與前述任務準確率屬於不同量測，
  無法用其中一項分數推出另一項分數或正式名次。receipt 記錄 coherence source revision
  `e18733694623aa93058e279c3534f8b8e2edefa3`；完整請求及回應 cache 一併保留。
- native regression 使用 repository 原本八個案例，包含文字、圖片、音訊、混合、長文字及
  多問題請求。這是推論差異檢查，不是新的品質 benchmark。

目前候選 cache 與 decoder compile 都維持顯式 opt-in。native 比較會逐題檢查所有
probability labels、每個機率值、argmax 及輸出 schema，依模態列出最大絕對機率差異。
沒有事先宣告的數值 tolerance 不會在看到結果後補成通過標準。
native 的 eager → cached → compiled 單次呼叫包含 shape 與 compilation 的影響，
不能據此宣稱速度提升；candidate cache 的獨立交錯 timing 實驗另由 campaign audit 處理。

`independent-native-regression.json` 另比較相同逐題 prompt 的 sequential reference 與
independent batching，直接使用八案例原始的 `case.request`，保留其圖片及原生音訊。
它逐一記錄 29 個問題所有機率與 label，並檢查 batch sizes、token counts 及 independent
執行宣告；right-padding masks 由固定版本的 helper 實作，回應本身不包含 masks。
BF16 的不同 GEMM batch shape 可能產生數值差異；
封存保留最大 drift 與變動 label，不套用事後 tolerance，也不宣稱與 causal multislot 語義等價。

## 已重算的 native／compile 結果

run02 的 `s1-compiled/native-regression.json` 包含八案例、29 題與 59 個候選機率。
candidate cache 與 eager 的全部機率一致，29/29 labels 一致。compiled 保留 29/29 labels，
但機率並非完全一致；最大差異為 0.0086020231（0.8602 個百分點）。

| 模態 | 問題數／候選機率數 | cache 最大機率差異 | compile 最大機率差異 | compile label 一致 |
| --- | ---: | ---: | ---: | ---: |
| 文字（五案例） | 23／47 | 0 | 0.0086020231 | 23／23 |
| 圖片 | 2／4 | 0 | 0.0080319643 | 2／2 |
| 音訊 | 2／4 | 0 | 0.0019716620 | 2／2 |
| 混合 | 2／4 | 0 | 0.0070635676 | 2／2 |

public231 的 raw hashes、requests、完整 population 與官方 scoring 已獨立重算。
baseline 與 compiled 都是 197/231（85.28%），但只有 227/231 個 argmax 一致：
兩題由對變錯，兩題由錯變對。公開 population 的最大機率差異達 0.507617265
（50.7617 個百分點）；不能因 aggregate accuracy 相同就宣稱 compiled 等價。
run02 四個 cells 的 campaign、runner 及 58 個 S1 Python source hashes 都匹配封存版本。

compiled 的公開首個呼叫耗時 363.8646552 秒，整個 231-case population 的 p50 為
50.7335 ms；baseline 首呼叫為 1.493381595 秒、population p50 為 88.8564 ms。
這些是不同 cell 的依序測量，包含不同 compilation／shape warming 狀態，
不能解讀成受控配對實驗的速度提升。原始首呼叫仍保留，不從 timing population 移除。
逐題差異及來源核對結果保留在 ZIP 的 `audits/compiled-native-audit-*.json`。

run03 的 independent 補充檢查同樣包含八案例、29 題與 59 個候選機率。
sequential reference 與 independent batching 的 29/29 個 argmax 一致；最大機率差異為
0.0063006282（0.6301 個百分點）。兩個 variants 共 58 份 answers 的完整欄位、id/type、
候選 labels、正規化機率、choice/level、noul、confidence/margin 及 score 都已重算核對。

| 模態 | 問題數／候選機率數 | independent 最大機率差異 | label 一致 |
| --- | ---: | ---: | ---: |
| 文字（五案例） | 23／47 | 0.0063006282 | 23／23 |
| 圖片 | 2／4 | 0.0020535067 | 2／2 |
| 音訊 | 2／4 | 0.0019716620 | 2／2 |
| 混合 | 2／4 | 0.0000000149 | 2／2 |

run03 independent 的 public231 與 run02 independent 主結果逐題核對：231/231 份 raw
requests、responses 及完整 raw JSON payload 相等，849 個機率全相等，最大差異為 0。
兩輪皆為 197/231（85.28%）；這個重現結果不包含新的 coherence 主結果，也沒有配對速度結論。
逐題資料與 schema 核對記錄保留在 ZIP 的 `audits/independent-native-audit-*.json`。
run03 兩個 cells 的 campaign、runner 及 58 個 S1 Python source hashes 皆為 60/60 匹配。

## 不可覆寫的封存檔

[archive_jevbench_campaign.py](../../../../scripts/archive_jevbench_campaign.py) 已產生
[raw-evidence.zip](raw-evidence.zip) 及 [raw-evidence.manifest.json](raw-evidence.manifest.json)。
ZIP 共 **8,219 個 members**，未壓縮總量 **28,970,101 bytes**，ZIP 大小 **10,932,994 bytes**。
全部 members 的 SHA256、size 與 CRC 均通過重讀核對；65 個 frozen package 檔案也與
commit `501c8a466fa6e694b184283d0574526a500b1655` 原始 bytes 相等。

- ZIP SHA256：`b85b4471d655a05e5fc682d2d22c157ae86af9b42a05a926e2bd78655223f53b`
- manifest：2,127,644 bytes；SHA256 `5cea21069490c69b59f8b9f12fd76062aeb2ed3b395a0ba3fe6a86d5d8d7b44d`
- 三輪共 1,848 份 public raw files 及 6,240 份完整 coherence cache files。

ZIP 使用固定時間
`1980-01-01T00:00:00`、sorted members、0644 mode 與 Deflate level 9。
相同輸入與相同 ZIP writer 會產生相同 bytes；每個 member 的 SHA256 與大小都列在外部
manifest，ZIP 本身亦有 SHA256。manifest 放在 ZIP 外，避免 self-hash 循環。

ZIP 保留：

- run01 的 S1 baseline、candidate cache 實驗、coherence report/cache、OneJev setup failure。
- run02 的 OneJev、Decider、S1 independent、S1 compiled 所有原始檔案。
- run03 的 Jev Omni 完整 public/coherence，以及 S1 independent 補充 public/native regression。
  這個補充 receipt 的 `coherence_requested=false`，不會再重跑 mini。它補上媒體驗證，
  不取代 run02 的 independent 完整 public/coherence 主結果。
- 各 run 的 Modal stdout/stderr log、receipt、public raw requests/responses/errors、records、summary、manifest。
- `executed-public01/02/03` 的實際 campaign/runner/unified bytes；`.py` 僅加 `.txt` 副檔名，不轉換內容。
- `package/` 內固定 commit `501c8a466fa6e694b184283d0574526a500b1655` 的整個 `src/s1`、
  `pyproject.toml`、`uv.lock`、build 所需的 `README.md` 與 MIT LICENSE。`src/s1/*.py` 同樣保留為 `.py.txt`。
- 同一 commit 的 `configs/benchmarks/models.toml`、`examples/benchmarks/mps.jsonl` 及
  `scripts/prepare_mlx_validation.py`；這些提供 Jev Omni registry、原始 native cases 與 checkpoint identity helper。
- 八個 native inputs、benchmark pins、BenchmarkHeaven LICENSE，以及 coherence LICENSE/NOTICE。
- `tools/` 另保存封存當下的 offline audit/archive tools。這組 bytes 與各 model cell 的
  `executed-public*/` 分開標示，不能當成先前雲端執行版本的證明。
- `audits/` 保留這輪 CPU read-only 重算的 native／public comparison；不修改雲端原始結果。
- `runtime-current/` 另外保留當前 CLI/backend wrapper wheel、原始 build manifest
  及明列選取範圍的 `archive-selection.json`。這組 build 的 source commit 為
  `1b581e90d4458cb658e565bc608990e8dee6174b`，不屬於較早雲端執行的 `501c8a4` package。
  wheel 為 195,197 bytes、SHA256 `8c16a6536480d6632d5f5478989ba1920d6ec2568eafe91647df2cb8591b38a1`；
  四個 wrapper/helper/unified source members 另對 Git commit 核對。文件重複的 18 MB sdist
  不收入 ZIP；原始 build manifest 的 sdist hash 仍完整保留，selection 明示未收入。
  GPU HTTP smoke 尚未執行，不能把 callable 評測或 wheel source 核對稱為新 HTTP runtime 已驗證。

模型權重不放進 ZIP；receipt 保存已執行 checkpoint 的 immutable revision、weight hashes、
套件版本與硬體。repository 的 `uv.lock` 不等於 Modal image 的已安裝環境，實際版本必須
讀各 receipt 及該次 campaign source，不能只用本機 lockfile 推定雲端版本。

run01 的舊 receipt 只記 runner hash，沒有 campaign 或整個 S1 package 的 execution hash。
其保留 source bytes 已對照當時另存 snapshot，屬於較弱的來源證明；manifest 明示
`not_recorded_in_receipt`。run02/run03 有記錄的 source hashes 都逐一核對，不會用今天的
工作目錄取代舊執行版本。整體 source proof 仍須區分「bytes 已封存」及「執行 receipt 有記錄」。

原始 artifacts 不會修改。目標 ZIP 或 manifest 任一已存在，archiver 都會拒絕寫入。
它也拒絕 symlink、權重檔、未完成且沒有 receipt 的 model cell，以及封存途中變動的 evidence。

```bash
python scripts/archive_jevbench_campaign.py
```

## 維護者驗證

取得 ZIP 及 manifest 後，先驗證 ZIP hash、完整 member set、每個 member hash 與 size：

```python
import hashlib, json, zipfile
from pathlib import Path

manifest = json.loads(Path("raw-evidence.manifest.json").read_text())
payload = Path("raw-evidence.zip").read_bytes()
assert hashlib.sha256(payload).hexdigest() == manifest["archive"]["sha256"]
assert len(payload) == manifest["archive"]["size_bytes"]
with zipfile.ZipFile("raw-evidence.zip") as bundle:
    assert bundle.testzip() is None
    assert bundle.namelist() == [row["path"] for row in manifest["members"]]
    for row in manifest["members"]:
        raw = bundle.read(row["path"])
        assert len(raw) == row["size_bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
```

在新的目錄解壓縮後，把 `.py.txt` 還原成 `.py` 即可取得保留的 source。
`package/` 保留可建置的原專案結構。新增的公開評測 config 位於 ZIP 根目錄的
`configs/benchmarks/jevbench-public.json`；在重跑 model cell 前，需複製到
`package/configs/benchmarks/`，並將該次 campaign/runner 還原到它們原本的
`apps/modal/jevbench_campaign.py` 與 `scripts/run_jevbench_public.py` 路徑。
以固定 revision 取得上游 public tasks/scoring，再用
[summarize_jevbench_campaign.py](../../../../scripts/summarize_jevbench_campaign.py)
獨立重算公開結果；請同時保留兩個上游專案的授權及 NOTICE。
public task population、coverage、schema validity、invalid responses、calibration、coherence、
timing scope 與成本缺值都需一起閱讀。單看 headline accuracy 無法證明正式上榜資格。

[完整 campaign audit](campaign-audit.md) 及 [其 JSON](campaign-audit.json) 記錄獨立重算結果。
本輪全量 CPU tests 為 1,240 passed、3 skipped，Ruff check／format 及 diff check 均通過；
archiver 的 focused tests 另有 20 passed。這些本機驗證不取代未執行的 GPU HTTP smoke。

[評測方案](../../../jevbench-plan.md) 記錄 protocols 與外部模型範圍；
[提交資料草稿](submission-draft.md) 列出等待維護者確認的事項。草稿仍未寄送或提交。
