"""app.py — 不動產謄本智慧解析系統（可上線版）"""
import hashlib
import hmac
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

import core

st.set_page_config(page_title="不動產謄本智慧解析系統", page_icon="🏡", layout="wide")

# 新版 Streamlit 以 width="stretch" 取代 use_container_width
_ver = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
WIDE = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}

st.markdown(
    """
<style>
.block-container{padding-top:2rem;max-width:1200px}
.hero h1{font-size:1.7rem;margin:0;background:linear-gradient(90deg,#60a5fa,#a78bfa);
-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{opacity:.7;margin:.25rem 0 1rem;font-size:.9rem}
div[data-testid="stMetric"]{background:rgba(128,128,128,.08);border:1px solid rgba(128,128,128,.2);
padding:10px 14px;border-radius:12px}
.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:600;min-height:44px}
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 設定讀取與登入
# ==========================================
def secret(name, default=""):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get(name, default)


def get_users():
    try:
        if "users" in st.secrets:
            return {str(k): str(v) for k, v in st.secrets["users"].items()}
    except Exception:  # noqa: BLE001
        pass
    return {}


USERS = get_users()
SHARED_PASSWORD = str(secret("APP_PASSWORD"))
AUTH_ENABLED = bool(USERS or SHARED_PASSWORD)
DAILY_LIMIT = int(secret("DAILY_TOKEN_LIMIT", 0) or 0)


def login_gate():
    if not AUTH_ENABLED:  # 未設定密碼＝本機開發模式
        st.session_state.setdefault("user", "local")
        return
    if st.session_state.get("user"):
        return
    st.markdown(
        '<div class="hero"><h1>🏡 不動產謄本智慧解析系統</h1><p>請先登入</p></div>',
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid, st.form("login"):
        name = st.text_input("使用者名稱")
        pwd = st.text_input("存取密碼", type="password")
        ok = st.form_submit_button("登入", type="primary", **WIDE)
    if ok:
        name = name.strip()
        if USERS:
            valid = name in USERS and hmac.compare_digest(USERS[name], pwd)
        else:
            valid = bool(name) and hmac.compare_digest(SHARED_PASSWORD, pwd)
        if valid:
            st.session_state["user"] = name
            st.rerun()
        time.sleep(1)  # 減緩暴力猜測
        with mid:
            st.error("帳號或密碼錯誤")
    st.stop()


login_gate()
user = st.session_state["user"]


def logout():
    st.session_state.clear()


def reset_all():
    st.session_state["reset_n"] = st.session_state.get("reset_n", 0) + 1
    st.session_state.pop("view", None)


def friendly(e: Exception) -> str:
    m = str(e)
    if re.search(r"API key|API_KEY_INVALID|\b401\b|\b403\b|PERMISSION_DENIED", m, re.I):
        return "API 金鑰無效或權限不足"
    if "404" in m or "NOT_FOUND" in m:
        return "找不到模型，請到側邊欄「進階設定」確認模型名稱"
    if "429" in m or "RESOURCE_EXHAUSTED" in m:
        return "API 額度或速率已達上限，請稍後再試或更換金鑰"
    return m[:300]


@st.cache_data(show_spinner=False, max_entries=32, ttl=3600)
def cached_extract(data: bytes, precise: bool):
    return core.extract_pdf(data, precise)


# ==========================================
# 側邊欄
# ==========================================
with st.sidebar:
    st.markdown(f"👤 **{user}**")
    if AUTH_ENABLED:
        st.button("登出", on_click=logout, **WIDE)

    st.header("⚙️ 標的設定")
    category = st.selectbox(
        "不動產標的類型", ["土地 (Land)", "建物 (Building)", "土地兼建物 (Land & Building)"]
    )
    zoning = st.selectbox(
        "使用分區 / 地目",
        [
            "都市計畫內 - 農地（農業區/保護區）",
            "都市計畫外 - 農牧用地/林業用地",
            "都市計畫內 - 建地（住宅/商業/工業區）",
            "都市計畫外 - 甲種/乙種/丙種建築用地",
            "其他分區（公共設施用地、山坡地保育區等）",
        ],
    )

    st.divider()
    st.subheader("📊 我的用量")
    today_tokens, total_tokens = core.usage_summary(user)
    st.metric("今日 Token", f"{today_tokens:,}", help=f"每日上限 {DAILY_LIMIT:,}" if DAILY_LIMIT else None)
    st.metric("累計 Token", f"{total_tokens:,}")
    if DAILY_LIMIT:
        st.progress(min(1.0, today_tokens / DAILY_LIMIT), text=f"每日額度 {today_tokens:,} / {DAILY_LIMIT:,}")

    with st.expander("🔧 進階設定"):
        st.text_input(
            "自備 Gemini API Key（僅存於本次連線）",
            type="password",
            key="own_key",
            help="填入後使用自己的額度，不受每日上限限制。",
        )
        model = st.text_input("模型名稱", value=str(secret("GEMINI_MODEL", core.DEFAULT_MODEL)))
        precise = st.toggle("精準版面模式（較慢）", value=False, help="保留 PDF 欄位對齊；一般電子謄本不需開啟。")
        vision = st.toggle("加強視覺核對（較慢、較準）", value=False, help="把含所有權部、住址的頁面轉成圖片一併送審。")
        mask = st.toggle("送出前遮蔽身分證後六碼", value=True)

# ==========================================
# 主畫面
# ==========================================
st.markdown(
    '<div class="hero"><h1>🏡 不動產謄本智慧解析系統</h1>'
    "<p>多模組並行解析 ｜ 結果即時導航 ｜ 手機可直接開啟</p></div>",
    unsafe_allow_html=True,
)

# ---- 步驟一：資料 ----
st.markdown("#### 1️⃣ 提供謄本資料")
n = st.session_state.setdefault("reset_n", 0)
cu, cp = st.columns([3, 2])
with cu:
    files = st.file_uploader(
        "拖曳或選擇謄本檔案（PDF / TXT，可多檔）",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"up_{n}",
    )
with cp:
    pasted = st.text_area("或貼上謄本文字", height=140, key=f"paste_{n}", placeholder="沒有檔案時可直接貼文字…")

docs = []
if files and len(files) > 20:
    st.warning("一次最多處理 20 個檔案，已取前 20 個。")
    files = files[:20]
for f in files or []:
    data = f.getvalue()
    if f.name.lower().endswith(".pdf"):
        try:
            pages_text, scanned = cached_extract(data, precise)
        except Exception as e:  # noqa: BLE001
            st.error(f"讀取 {f.name} 失敗：{e}")
            continue
        docs.append(
            dict(name=f.name, kind="pdf", bytes=data, pages=len(pages_text), pages_text=pages_text, scanned=scanned)
        )
    else:
        docs.append(dict(name=f.name, kind="txt", pages=1, text=core.decode_text(data), scanned=[]))

parts = []
for d in docs:
    if d["kind"] == "pdf":
        parts.append(core.build_doc_text(d["name"], d["pages_text"]))
    else:
        parts.append(f"---【檔案：{d['name']}】---\n{core.clean_text(d['text'])}")
if pasted.strip():
    parts.append(f"---【手動貼上之謄本內容】---\n{pasted.strip()}")
final_text = "\n\n".join(parts)
if mask:
    final_text = core.mask_ids(final_text)

has_input = bool(docs or pasted.strip())
if has_input:
    total_pages = sum(d["pages"] for d in docs) + (1 if pasted.strip() else 0)
    scanned_n = sum(len(d["scanned"]) for d in docs)
    m1, m2, m3 = st.columns(3)
    m1.metric("📁 檔案", f"{len(docs) + (1 if pasted.strip() else 0)} 份")
    m2.metric("📄 總頁數", f"{total_pages} 頁")
    m3.metric("🔤 文字量", f"約 {len(final_text):,} 字")
    if scanned_n:
        st.warning(f"有 {scanned_n} 頁含文字層讀不到的欄位（如住址）或為掃描檔，將自動加送頁面圖片辨識，速度會稍慢。")
    if total_pages > 80:
        st.warning("總頁數偏多，解析時間與費用會明顯增加，建議分批處理。")
    with st.expander("🔍 檢視送出的精煉文字"):
        st.text_area("預覽", value=final_text, height=200, label_visibility="collapsed")

# ---- 步驟二：模組 ----
st.markdown("#### 2️⃣ 選擇解析模組")
choice = st.pills(
    "可多選；選到的模組會同時並行解析，總時間約等於最慢的那一個",
    options=list(core.MODULES),
    format_func=lambda k: core.MODULES[k]["label"],
    selection_mode="multi",
    default=["route"],
    key="mods",
)
keys = [k for k in core.MODULES if k in (choice or [])]

# ---- 步驟三：執行 ----
c_run, c_clear = st.columns([4, 1])
run = c_run.button("🚀 開始解析", type="primary", disabled=not (has_input and keys), **WIDE)
c_clear.button("🗑️ 清空", on_click=reset_all, **WIDE)

if run:
    own_key = st.session_state.get("own_key", "").strip()
    api_key = own_key or str(secret("GEMINI_API_KEY"))
    if not api_key:
        st.error("尚未設定 Gemini API Key：請管理員在 Secrets 設定 GEMINI_API_KEY，或於側邊欄「進階設定」貼上自己的金鑰。")
    elif not own_key and DAILY_LIMIT and today_tokens >= DAILY_LIMIT:
        st.error("今日額度已用完，請明天再試，或於「進階設定」使用自己的 API Key。")
    else:
        t_start = time.time()
        cache = st.session_state.setdefault("mod_cache", {})
        needs_img = any(d["kind"] == "pdf" and (d["scanned"] or vision) for d in docs)

        h = hashlib.sha256()
        for s in (final_text, category, zoning, model, str(vision), str(mask)):
            h.update(s.encode("utf-8"))
            h.update(b"\0")
        if needs_img:
            for d in docs:
                if d["kind"] == "pdf":
                    h.update(d["bytes"])
        sig = h.hexdigest()[:16]

        todo = [k for k in keys if f"{sig}:{k}" not in cache]  # 已解析過的模組直接沿用
        errors = {}
        if todo:
            with st.status(f"解析中（{len(todo)} 個模組並行）…", expanded=True) as status:
                images = []
                if needs_img:
                    st.write("🖼️ 擷取需視覺辨識的頁面…")
                    try:
                        images = core.collect_images(docs, vision)
                    except Exception as e:  # noqa: BLE001
                        st.write(f"⚠️ 頁面轉圖失敗，改用純文字：{e}")
                client = core.make_client(api_key)
                st.write(f"⚡ 已同時送出 {len(todo)} 個解析請求…")
                with ThreadPoolExecutor(max_workers=len(todo)) as ex:
                    futs = {
                        ex.submit(core.run_module, client, model, k, category, zoning, final_text, images): k
                        for k in todo
                    }
                    for fut in as_completed(futs):
                        k = futs[fut]
                        label = core.MODULES[k]["label"]
                        try:
                            r = fut.result()
                            cache[f"{sig}:{k}"] = r
                            core.log_usage(user, model, k, r["p"], r["o"])
                            st.write(f"✅ {label} · {r['secs']:.1f} 秒")
                        except Exception as e:  # noqa: BLE001
                            errors[k] = friendly(e)
                            st.write(f"❌ {label}")
                status.update(label="解析完成" if not errors else "部分模組失敗", state="complete" if not errors else "error")
            while len(cache) > 20:  # 限制單一連線的記憶體占用
                cache.pop(next(iter(cache)))

        st.session_state["view"] = {
            "keys": keys,
            "results": {k: cache[f"{sig}:{k}"] for k in keys if f"{sig}:{k}" in cache},
            "errors": errors,
            "elapsed": time.time() - t_start,
            "reused": [k for k in keys if k not in todo],
            "model": model,
        }
        st.rerun()  # 重新整理側邊欄用量


# ==========================================
# 結果顯示
# ==========================================
def render_module(k: str, r: dict):
    if r["truncated"]:
        st.warning("⚠️ 輸出達長度上限，結果可能不完整；建議減少檔案數量後重新解析。")
    if k == "route":
        df = r.get("df")
        if df is not None and not df.empty:
            show = core.route_display_df(df)
            a, b, c = st.columns(3)
            a.metric("拜訪筆數", len(show))
            b.metric("所有權人", show["所有權人"].nunique())
            c.metric("可導航地址", int(show["導航"].notna().sum()))
            st.caption("點「📍 導航」直接開啟 Google Maps；標示「謄本未載」的地址需另行查詢。")
            st.dataframe(
                show,
                hide_index=True,
                column_config={
                    "導航": st.column_config.LinkColumn("導航", display_text="📍 導航"),
                    "戶籍/通訊地址": st.column_config.TextColumn(width="large"),
                    "備註與建議": st.column_config.TextColumn(width="large"),
                },
                **WIDE,
            )
            st.download_button(
                "📥 下載拜訪名單 (.csv)",
                data=show.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"拜訪名單_{core.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
            )
            return
        st.warning("拜訪表格解析失敗，以下為模型原始輸出。")
        st.code(r["text"])
        return
    st.markdown(r["text"])


view = st.session_state.get("view")
if view:
    st.divider()
    st.markdown("### 📑 解析結果")
    note = f"⏱ 本次耗時 {view['elapsed']:.1f} 秒 ｜ 模型：{view['model']}"
    if view["reused"]:
        note += " ｜ 已沿用先前結果：" + "、".join(core.MODULES[k]["label"] for k in view["reused"])
    st.caption(note)

    for k, msg in view["errors"].items():
        st.error(f"{core.MODULES[k]['label']} 失敗：{msg}（再按一次「開始解析」只會重跑失敗的模組）")

    ok_keys = [k for k in view["keys"] if k in view["results"]]
    if ok_keys:
        stamp = core.now().strftime("%Y%m%d_%H%M")
        d1, d2, _ = st.columns([1.3, 1.3, 2])
        d1.download_button(
            "📥 手機版網頁 (.html)",
            data=core.build_html(view),
            file_name=f"產權解析報告_{stamp}.html",
            mime="text/html",
            **WIDE,
        )
        d2.download_button(
            "📥 文字報告 (.md)",
            data=core.build_markdown(view),
            file_name=f"謄本解析報告_{stamp}.md",
            mime="text/markdown",
            **WIDE,
        )
        tabs = st.tabs([core.MODULES[k]["label"] for k in ok_keys])
        for tab, k in zip(tabs, ok_keys):
            with tab:
                render_module(k, view["results"][k])

st.caption("本系統輸出僅供參考，實際權利狀態以地政機關登記為準；稅費試算以稽徵機關核定為準。")
