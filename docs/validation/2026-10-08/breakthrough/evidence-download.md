# 完整證據包重組

本輪完整 ZIP 為 116,641,459 bytes，含 26,135 個檔案。為便於分享，Git 保存兩個依序串接的分片；完整 ZIP 留在本機，不重複提交。這是既有封存檔的逐位元分割，報告、原始資料與校驗狀態都沒有重新產生或修改，也未上傳或發布。

| 分片 | Bytes | SHA256 |
| --- | ---: | --- |
| [part01](newcloud-evidence.zip.part01) | 83,886,080 | `9467d6866d5617e0dcfc905538f842c2cd42a37731d1c9344efd7c869fb81a17` |
| [part02](newcloud-evidence.zip.part02) | 32,755,379 | `05fbf0078eeb7b5a84842cb430b12f260a76f00ff1a5988d6d94a1d26d128cc8` |

在此目錄執行以下 Python 3 指令。它先核對分片，依序重組，再核對完整 ZIP 的大小、SHA256 及 CRC；已存在的完整 ZIP 也會接受核對。

```sh
python3 - <<'PY'
from pathlib import Path
import hashlib, json, zipfile

spec = json.loads(Path("newcloud-evidence.parts.json").read_text())
payloads = []
for part in spec["parts"]:
    payload = Path(part["filename"]).read_bytes()
    assert len(payload) == part["size_bytes"], part["filename"]
    assert hashlib.sha256(payload).hexdigest() == part["sha256"], part["filename"]
    payloads.append(payload)
archive = spec["archive"]
combined = b"".join(payloads)
assert len(combined) == archive["size_bytes"]
assert hashlib.sha256(combined).hexdigest() == archive["sha256"]
target = Path(archive["filename"])
if target.exists():
    assert target.read_bytes() == combined
else:
    with target.open("xb") as output:
        output.write(combined)
with zipfile.ZipFile(target) as evidence:
    assert len(evidence.infolist()) == 26135
    assert evidence.testzip() is None
print("Verified:", target)
PY
```

完整 ZIP 的 SHA256 為 `53cc9767dbffefd29170399e3de9ec538cbfd71d86d3e5ed4251d0c48ae53218`。機器可讀的分片順序與雜湊見 [分片 manifest](newcloud-evidence.parts.json)；每個 ZIP member 的大小、SHA256 與來源見 [完整 manifest](newcloud-evidence.manifest.json)。完整 manifest 的 SHA256 為 `085ed0c9b335795e935a979adc24f64c2059247036edf0ce07ba3499eb27aca6`。

需要逐檔重驗時，在專案根目錄執行：

```sh
uv run --frozen python scripts/archive_breakthrough.py --verify docs/validation/2026-10-08/breakthrough/newcloud-evidence.zip
```

此分享說明是封存完成後的補充，不在 ZIP 內。ZIP 的 `reports/README.md` 與 `reports/results.md` 保留封存當時報告；[結果報告](results.md) 說明六組完整稽核、兩組機率連結未完成，以及 CPU03 上傳遭自動審核拒絕的狀態。
