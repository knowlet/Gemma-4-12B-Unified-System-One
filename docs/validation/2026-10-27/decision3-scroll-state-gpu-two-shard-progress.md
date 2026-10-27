# Decision3 scroll-state：兩批 GPU 測試進度

日期：2026-10-27

已驗收的 equal 訓練涵蓋 273 筆 operation 與 65 筆 click_target。公開 commit `f20be27` 已記錄[兩組訓練結果](../2026-10-26/decision3-scroll-state-two-arm-fit-results.md)。

兩批 GPU 測試各完成 27 次原生模型呼叫，合計 54 次，涵蓋全部 48 筆原始輸入、兩個 zero controls 與四個 invented controls。全部 48 筆原始輸入的 shape 已於啟動前校驗。兩次原 controller 的原始工作階段均已終止，exit code 均為 0，經過時間分別為 66.024 與 69.016 秒。

兩批的獨立 FP64 逐頭重算及完整 48 筆原始輸入 union 檢查均已回報成功；aggregate 階段的原生模型呼叫數為 0。已報告的 baseline 最大誤差為 `2.597254158409612e-7`，candidate 最大誤差為 `3.2994945564635003e-7`。

| 重算範圍 | 每個 branch 的 heads | 每個 branch 的 legal positions |
| --- | ---: | ---: |
| 48 筆原始輸入 | 192 | 1,954 |
| 含全部 controls | 208 | 3,064 |

兩個 branches 合計進行 6,128 次 legal-position 比較。

完整獨立驗收仍未完成。第一版 replay 曾因驗證 runtime 來源路徑時，將 workspace overlay 誤認為應位於 vendor 安裝位置而失敗，該版本與紀錄均保留。後續來源審查又發現，審核器的 final rehash 尚未涵蓋兩個已選定的 command stdout／stderr 檔案，因此先前 aggregate 的「全輸入身分已重驗」宣稱不完整。原報告已保留，正修正第三版並重新覆核；上述數值重算結果不構成完整獨立驗收通過。

合成控制測試結果如下，僅代表控制測試通過：

- GPU qualifier：21/21。
- 第一案例整合：18/18。
- 獨立 auditor：11 + 6 + 9 項。

新 candidate 尚未執行真實瀏覽器測試，也尚未執行 JevBench coherence 與 Benchmark Heaven 兩套評測；目前均無分數，不宣稱可用性改善或優於其他模型。
