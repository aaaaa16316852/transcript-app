"""core.py — 與 Streamlit 無關的核心邏輯（文字處理、提示詞、Gemini 呼叫、用量資料庫、報告輸出）"""
from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import time
import unicodedata
from contextlib import closing
from datetime import datetime, timedelta, timezone
from html import escape
from urllib.parse import quote

import pandas as pd
import pdfplumber
from google import genai
from google.genai import types

TZ = timezone(timedelta(hours=8))  # 台灣時間（無日光節約）
DEFAULT_MODEL = "gemini-3.8-flash"
MAX_VISION_PAGES = 12
DB_PATH = os.environ.get("USAGE_DB", "usage.db")


def now() -> datetime:
    return datetime.now(TZ)


# ==========================================
# 1. 文字處理
# ==========================================
# NFKC 轉不掉的部首補充區字元（地政網站的「民國」用的是 U+2EA0）
_RADICAL_FIX = str.maketrans({"\u2ea0": "民"})


def _norm(t: str) -> str:
    return unicodedata.normalize("NFKC", t).translate(_RADICAL_FIX)


_SPACES = re.compile(r"[ \t\u3000]{2,}")
# 台灣身分證：1 英文 + 1/2 + 8 碼數字；只留前四碼，其餘遮蔽
_ID = re.compile(r"(?<![A-Za-z0-9])([A-Z][12]\d{2})\d{6}(?![0-9])")
VISION_KEYWORDS = ("所有權部", "登記名義人", "權利範圍", "住址", "使用分區", "他項權利部")


def clean_text(t: str) -> str:
    """保留所有非空白行（不再丟棄短行，避免漏掉單獨成行的持分如 1/3），僅壓縮多餘空白。"""
    lines = []
    for line in _norm(t).splitlines():
        line = _SPACES.sub("  ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def mask_ids(t: str) -> str:
    return _ID.sub(r"\1******", t)


def decode_text(b: bytes) -> str:
    for enc in ("utf-8-sig", "big5", "cp950"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="ignore")


_WATERMARK_SIZE = 30  # 正文約 9–12pt；浮水印字級遠大於此（實測 71.7pt）


def _needs_vision(text: str) -> bool:
    """文字層過少（掃描檔），或「住址」標籤後沒有內容（地政網站常把住址渲染成圖片）。"""
    if len(text.strip()) < 30:
        return True
    return any(re.search(r"住址\s*$", ln) for ln in text.splitlines())


def extract_pdf(data: bytes, precise: bool = False):
    """回傳 (每頁文字, 需視覺辨識的頁索引)。
    會先濾掉大字級浮水印，並做 NFKC 正規化。precise=True 使用版面模式，較慢但保留欄位對齊。"""
    pages_text, need_vision = [], []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages):
            page = page.filter(lambda o: o.get("object_type") != "char" or o.get("size", 0) < _WATERMARK_SIZE)
            t = _norm(page.extract_text(layout=precise) or "")
            if _needs_vision(t):
                need_vision.append(i)
            pages_text.append(t)
    return pages_text, need_vision


def build_doc_text(name: str, pages_text: list[str]) -> str:
    body = "\n".join(
        f"[第{i + 1}頁]\n{clean_text(t)}" for i, t in enumerate(pages_text) if t.strip()
    )
    return f"---【檔案：{name}（共 {len(pages_text)} 頁）】---\n{body}"


def collect_images(docs: list[dict], deep: bool) -> list:
    """掃描頁一律轉圖；deep=True 時，含關鍵字的頁面也轉圖供視覺核對。"""
    parts: list = []
    for d in docs:
        if d["kind"] != "pdf":
            continue
        idx = set(d["scanned"])
        if deep:
            idx |= {i for i, t in enumerate(d["pages_text"]) if any(k in t for k in VISION_KEYWORDS)}
        if not idx:
            continue
        with pdfplumber.open(io.BytesIO(d["bytes"])) as pdf:
            for i in sorted(idx):
                if len(parts) >= MAX_VISION_PAGES:
                    return parts
                im = pdf.pages[i].to_image(resolution=130).original.convert("RGB")
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=80)
                parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
    return parts


# ==========================================
# 2. 提示詞與模組定義
# ==========================================
SYSTEM_PROMPT = (
    "你是台灣地政與不動產權利分析專家，負責解讀地政電子謄本。\n"
    "守則：\n"
    "1. 只能依據使用者提供的謄本資料（文字與附圖）作答；資料沒有的寫「謄本未載」，嚴禁推測或編造。\n"
    "2. 地號、面積、持分、金額、日期須逐字核對原文，不得四捨五入或改寫。\n"
    "3. 引用事項時註明來源頁碼或登記次序（例如：第2頁、登記次序0003）。\n"
    "4. 使用繁體中文，直接輸出結果，不要開場白、結尾客套或重複題目。\n"
    "5. 若附有頁面圖片，文字層缺漏或不一致的欄位（如住址）以圖片為準。"
)

_ROUTE_PROMPT = """任務：整理外勤拜訪名單。
只輸出一個 JSON 陣列（不要 Markdown、不要任何說明文字）。每個元素代表「一位所有權人 × 一筆標的（地號或建號）」，欄位固定如下：
- station：拜訪站別與區域，例如「第1站｜善化區文正里」；請依行政區、里鄰把相近地址排在一起，方便順路拜訪
- owner：所有權人稱謂（姓氏加「先生／女士」；公司、法人、機關保留全名）
- id_prefix：身分證字號前四碼（僅前四碼，其餘不得出現）
- address：謄本所載戶籍或通訊地址，須含完整門牌號碼與樓層；謄本只有路名或未載明時填「謄本未載」，嚴禁自行補全或猜測
- land_no：地號或建號
- area：面積（含單位）
- share：權利範圍（持分，如 1/3）
- reason：登記原因與登記日期
- note：備註與拜訪建議（如共有人關係、抵押、限制登記；無則空字串）
所有欄位皆為字串。"""

_RISK_PROMPT = """任務：風險紅綠燈評估。
先給一行「總評」（🔴高風險／🟡需留意／🟢單純 + 一句理由），再用 Markdown 表格逐項列出：
| 項目 | 燈號 | 依據（頁碼／登記次序） | 建議處理 |
必查項目：共有（產權僵局）、抵押權（權利人、設定金額、債務人、存續期間）、查封／假扣押／假處分／預告登記等限制登記、地上權／典權／租賃等他項權利、使用分區與地目限制。
沒有的項目也要列出，燈號標 🟢 並註明「謄本未載」。"""

_INHERIT_PROMPT = """任務：繼承與關鍵權利人分析。
1. 共有人結構表：| 權利人（稱謂） | 持分 | 持分占比 | 取得原因 | 是否可能未辦繼承 |
   「是否可能未辦繼承」僅依謄本是否標註已歿或登記原因為繼承判斷；無法確定者寫「需戶籍資料確認」。
2. 關鍵權利人：持分最大者是誰、哪些組合可超過二分之一或三分之二持分，並說明對處分、出售或整合開發的影響。
3. 整合策略：建議的接洽順序與談判重點，各一至兩句。"""

_TAX_PROMPT = """任務：依指定分區（__ZONING__）整理稅費評估。
1. 先列適用稅目（如土地增值稅、房地合一稅、契稅、印花稅，以及地價稅／房屋稅等持有稅負），各以一句說明適用與否及原因。
2. 逐稅目給出「計算公式」與稅率級距（以現行法規為準）。
3. 謄本無法提供的數值（公告現值、取得成本、預計售價、持有年限等）列為「需補充」，不得自行假設數字；資料齊全才給試算結果。
4. 特別提示該分區常見的免稅、減免或限制（例如農地承受資格、農業使用證明）。
結尾加一行：「以上為概算，實際稅額以稽徵機關核定為準。」"""

_TIMELINE_PROMPT = """任務：歷史時間軸與持分脈絡。
以 Markdown 表格由舊到新列出：| 日期 | 登記次序 | 登記原因 | 權利人變動（前→後） | 持分變化 | 備註 |
表格後以 3–5 點條列：目前分區／地目狀態、關鍵移轉節點、最近一次移轉現值或原規定地價（作為增值基期）、持分是否有分割或合併。"""

# 顯示順序即此處順序
MODULES: dict[str, dict] = {
    "route": dict(label="📍 外勤拜訪路徑", prompt=_ROUTE_PROMPT, max_tokens=8192, json=True),
    "risk": dict(label="🚦 風險紅綠燈", prompt=_RISK_PROMPT, max_tokens=3000, json=False),
    "inherit": dict(label="👪 繼承與共有人", prompt=_INHERIT_PROMPT, max_tokens=3000, json=False),
    "tax": dict(label="💰 稅費試算", prompt=_TAX_PROMPT, max_tokens=3000, json=False),
    "timeline": dict(label="🕰️ 歷史時間軸", prompt=_TIMELINE_PROMPT, max_tokens=4000, json=False),
}


def build_prompt(key: str, category: str, zoning: str, text: str) -> str:
    head = f"【標的類型】{category}\n【指定使用分區與地目】{zoning}\n\n"
    body = MODULES[key]["prompt"].replace("__ZONING__", zoning)
    data = f"\n\n【謄本資料】\n{text}" if text else "\n\n（無文字資料，請依附圖頁面辨識內容）"
    return head + body + data


# ==========================================
# 3. Gemini 呼叫（含重試；不碰任何 Streamlit 物件，可安全在執行緒中執行）
# ==========================================
_RETRYABLE = re.compile(r"429|500|503|504|RESOURCE_EXHAUSTED|UNAVAILABLE|overloaded|timed? ?out", re.I)


def make_client(api_key: str):
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=120_000))


def generate(client, model: str, contents, key: str) -> dict:
    spec = MODULES[key]
    cfg = dict(system_instruction=SYSTEM_PROMPT, temperature=0.1, max_output_tokens=spec["max_tokens"])
    if spec["json"]:
        cfg["response_mime_type"] = "application/json"

    thinking_off, attempt, delay = True, 0, 1.5  # 關閉思考以求速度；模型不支援時自動退回
    t0 = time.time()
    while True:
        kwargs = dict(cfg)
        if thinking_off:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        try:
            resp = client.models.generate_content(
                model=model, contents=contents, config=types.GenerateContentConfig(**kwargs)
            )
            break
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if thinking_off and re.search(r"thinking|budget", msg, re.I):
                thinking_off = False
                continue
            if attempt < 3 and _RETRYABLE.search(msg):
                attempt += 1
                time.sleep(delay)
                delay *= 2
                continue
            raise

    um = getattr(resp, "usage_metadata", None)
    p = (getattr(um, "prompt_token_count", 0) or 0) if um else 0
    o = (
        (getattr(um, "candidates_token_count", 0) or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
        if um
        else 0
    )
    truncated = False
    try:
        truncated = "MAX_TOKENS" in str(resp.candidates[0].finish_reason)
    except Exception:  # noqa: BLE001
        pass

    out = dict(text=resp.text or "", p=p, o=o, secs=time.time() - t0, truncated=truncated, df=None)
    if key == "route":
        try:
            out["df"] = parse_route(out["text"])
        except Exception:  # noqa: BLE001
            pass
    return out


def run_module(client, model, key, category, zoning, text, images) -> dict:
    prompt = build_prompt(key, category, zoning, text)
    contents = [*images, prompt] if images else prompt
    return generate(client, model, contents, key)


# ==========================================
# 4. 拜訪路徑：解析 JSON、產生 Google Maps 連結
# ==========================================
ROUTE_FIELDS = [
    ("station", "拜訪站別"),
    ("owner", "所有權人"),
    ("address", "戶籍/通訊地址"),
    ("land_no", "地號/建號"),
    ("area", "面積"),
    ("share", "持分"),
    ("reason", "登記原因"),
    ("id_prefix", "身分證前四碼"),
    ("note", "備註與建議"),
]


def parse_route(text: str) -> pd.DataFrame:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M).strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        # 輸出被截斷時，保留最後一個完整物件
        end = t.rfind("}")
        if not t.startswith("[") or end < 0:
            raise
        data = json.loads(t[: end + 1] + "]")
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    rows = [
        {k: str(r.get(k) or "").strip() for k, _ in ROUTE_FIELDS}
        for r in data
        if isinstance(r, dict)
    ]
    return pd.DataFrame(rows, columns=[k for k, _ in ROUTE_FIELDS])


def maps_url(addr: str) -> str | None:
    a = (addr or "").strip()
    if not a or "未載" in a:
        return None
    return "https://www.google.com/maps/search/?api=1&query=" + quote(a)


def route_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=dict(ROUTE_FIELDS)).copy()
    out.insert(3, "導航", [maps_url(a) for a in df["address"]])
    return out


def route_markdown(df: pd.DataFrame) -> str:
    heads = [v for _, v in ROUTE_FIELDS]
    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for _, r in df.iterrows():
        cells = []
        for k, _ in ROUTE_FIELDS:
            v = str(r[k]).replace("|", "／").replace("\n", " ")
            if k == "address" and maps_url(v):
                v = f"[{v}]({maps_url(v)})"
            cells.append(v)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ==========================================
# 5. 用量資料庫（SQLite，逐人記錄）
# ==========================================
def _db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS usage ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, day TEXT, user TEXT, model TEXT, "
        "module TEXT, prompt_tokens INTEGER, output_tokens INTEGER)"
    )
    return conn


def log_usage(user: str, model: str, module: str, p: int, o: int) -> None:
    n = now()
    with closing(_db()) as conn, conn:
        conn.execute(
            "INSERT INTO usage (ts, day, user, model, module, prompt_tokens, output_tokens) VALUES (?,?,?,?,?,?,?)",
            (n.strftime("%Y-%m-%d %H:%M:%S"), n.strftime("%Y-%m-%d"), user, model, module, p, o),
        )


def usage_summary(user: str) -> tuple[int, int]:
    today = now().strftime("%Y-%m-%d")
    with closing(_db()) as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(CASE WHEN day=? THEN prompt_tokens+output_tokens END),0), "
            "COALESCE(SUM(prompt_tokens+output_tokens),0) FROM usage WHERE user=?",
            (today, user),
        ).fetchone()
    return int(row[0]), int(row[1])


# ==========================================
# 6. 報告輸出（Markdown / 手機版 HTML）
# ==========================================
def build_markdown(view: dict) -> str:
    out = [f"# 謄本智慧解析報告\n\n產生時間：{now():%Y-%m-%d %H:%M}（模型：{view.get('model', '')}）\n"]
    for k in view["keys"]:
        r = view["results"].get(k)
        if not r:
            continue
        out.append(f"\n## {MODULES[k]['label']}\n")
        out.append(route_markdown(r["df"]) if k == "route" and r.get("df") is not None else r["text"])
    return "\n".join(out)


def _md_to_html(text: str) -> str:
    try:
        import markdown

        # 先跳脫，避免謄本內容夾帶的 HTML／script 被直接輸出
        return markdown.markdown(escape(text, quote=False), extensions=["tables"])
    except Exception:  # noqa: BLE001
        return f"<pre>{escape(text)}</pre>"


def _route_cards(df: pd.DataFrame) -> str:
    cards = []
    for _, r in df.iterrows():
        url = maps_url(r["address"])
        if url:
            nav = (
                f'<a class="nav" href="{escape(url, quote=True)}" target="_blank" rel="noopener">'
                f'📍 導航　{escape(r["address"])}</a>'
            )
        else:
            nav = f'<div class="noaddr">地址：{escape(r["address"] or "謄本未載")}</div>'
        note = f'<div class="note">{escape(r["note"])}</div>' if r["note"] else ""
        cards.append(
            '<div class="card">'
            f'<div class="head"><b>{escape(r["station"])}</b>'
            f'<span>{escape(r["owner"])}　{escape(r["id_prefix"])}</span></div>'
            f'<div class="kv">地號／建號：{escape(r["land_no"])}　面積：{escape(r["area"])}　持分：{escape(r["share"])}</div>'
            f'<div class="kv">登記原因：{escape(r["reason"])}</div>{nav}{note}</div>'
        )
    return "".join(cards)


_HTML_CSS = """
:root{--bg:#f8fafc;--fg:#0f172a;--mute:#475569;--card:#fff;--line:#e2e8f0;--acc:#2563eb;--head:#1d4ed8}
@media(prefers-color-scheme:dark){:root{--bg:#0b0f19;--fg:#f1f5f9;--mute:#94a3b8;--card:#111827;--line:#1f2937;--acc:#60a5fa;--head:#93c5fd}}
*{box-sizing:border-box}body{margin:0;padding:14px;background:var(--bg);color:var(--fg);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang TC","Noto Sans TC",sans-serif;line-height:1.6}
h1{font-size:20px;margin:0 0 4px}h2{font-size:17px;color:var(--head);margin:22px 0 10px}
.sub{color:var(--mute);font-size:12px;margin-bottom:8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:10px}
.head{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}.head span{color:var(--mute);font-size:13px}
.kv{font-size:13px;color:var(--mute);margin-top:4px}.note{font-size:13px;margin-top:6px}
.nav{display:block;margin-top:10px;padding:12px;border-radius:10px;background:var(--acc);color:#fff;
text-decoration:none;font-weight:700;font-size:14px;word-break:break-all}.noaddr{margin-top:8px;color:#b45309;font-size:13px}
.md{overflow-x:auto}table{border-collapse:collapse;min-width:520px;width:100%;font-size:13px}
th,td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}th{background:var(--card)}
"""


def build_html(view: dict) -> str:
    parts = []
    for k in view["keys"]:
        r = view["results"].get(k)
        if not r:
            continue
        parts.append(f"<section><h2>{escape(MODULES[k]['label'])}</h2>")
        if k == "route" and r.get("df") is not None and not r["df"].empty:
            parts.append(_route_cards(r["df"]))
        else:
            parts.append(f'<div class="md">{_md_to_html(r["text"])}</div>')
        parts.append("</section>")
    return (
        "<!DOCTYPE html><html lang='zh-TW'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>產權與外勤解析報告</title><style>{_HTML_CSS}</style></head><body>"
        f"<h1>🏡 產權與外勤解析報告</h1><div class='sub'>{now():%Y-%m-%d %H:%M} ｜ 僅供參考，以地政機關登記為準</div>"
        + "".join(parts)
        + "</body></html>"
    )
