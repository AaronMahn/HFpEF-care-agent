# CARE-Agent — 评分说明

你会收到一个 HTML 文件：`rating_sheet_<你的姓名>.html`。

1. 在浏览器中双击打开（Chrome、Safari、Edge 均可）。
2. 每位患者（P-01 至 P-20）下方会展示 **6 个匿名模型**（标记为
   Model A / B / C / …），同一字母在不同患者间对应不同模型，避免品牌偏倚。
3. 如需临床背景，可点击"查看患者基础报告（无 LLM）"展开。对**每个**模型输出，按以下
   六个维度 0–2 评分：
   - 机制解释准确性
   - 风险驱动因素识别
   - 治疗建议合理性
   - 安全标记质量
   - 幻觉/错误推断（**反向计分** — 分越高代表幻觉越多）
   - 临床可读性
4. 可选填写"点评"。
5. **保存与续做（重要）。** 请先把压缩包**完整解压**到一个固定文件夹，不要直接在
   压缩包里打开 HTML。按推荐程度，有三种保存方式：
   - **最省心（推荐，需已装 Python）：** 双击随附的启动器——macOS 用
     **`Start_rating.command`**，Windows 用 **`Start_rating.bat`**。它会自动打开
     评分表，之后**每次打分都自动存进这个文件夹**（生成 `ratings_<姓名>_*.json`），
     下次再双击启动器即可**自动读回**，全程无需选文件、无需点任何按钮。评分时请
     **保持启动器的黑色窗口开着**，做完再关。
   - **Chrome / Edge 直接双击 HTML：** 打开后点右上角 **连接文件**，选一个保存位置，
     之后自动写入该文件；下次打开再点一次 **连接文件**、选同一个文件即可继续。
   - **兜底（Safari / Firefox）：** 进度暂存在当前浏览器里，换电脑/浏览器/移动文件后
     无法自动恢复。请每做一部分点 **导出 JSON**，下次用 **导入 / 继续** 载入。
6. 全部完成后，把文件夹里的 `ratings_<姓名>_*.json`（或 **导出 JSON** 下载的文件）回传。

谢谢。


---

# CARE-Agent — Rating Sheet (Instructions)

You'll receive a single HTML file: `rating_sheet_<your_name>.html`.

1. Double-click to open it in any modern browser (Chrome, Safari, Edge).
2. For each patient (P-01 to P-20), you'll see **6 anonymous model
   outputs** labelled **Model A / B / C / …**. The labels are re-randomised per
   patient so brand pattern recognition is impossible.
3. Read the patient's *base report* (click the "Show patient base report" disclosure
   if you need the clinical context), then for **each** model output rate the six
   dimensions on a 0–2 scale:
   - Mechanism accuracy
   - Risk-driver identification
   - Treatment-recommendation rationality
   - Safety-flag quality
   - Hallucination *(reverse-scored — higher = worse)*
   - Clinical readability
4. Add a comment if useful (optional).
5. **Saving & resuming (important).** First **fully extract** the zip to a fixed
   folder; do not open the HTML from inside the zip. Three ways, best first:
   - **Easiest (recommended, needs Python installed):** double-click the bundled
     launcher — **`Start_rating.command`** on macOS, **`Start_rating.bat`** on
     Windows. It opens your sheet and **auto-saves every answer into this
     folder** (`ratings_<your_name>_*.json`); next time just double-click the
     launcher again and it **auto-loads** — no file pickers, no buttons. Keep
     the launcher's black window **open** while rating; close it when done.
   - **Chrome / Edge by double-clicking the HTML:** click **Link file** (top
     right) once, choose a location, and answers auto-write to it. Next time
     click **Link file** again and pick the same file to continue.
   - **Fallback (Safari / Firefox):** progress stays only in this browser and
     won't restore elsewhere. Click **Export JSON** as you go and **Import /
     Resume** next time.
6. When finished, send back `ratings_<your_name>_*.json` from the folder (or the
   **Export JSON** download).

Thank you.
