"""
Multilingual Translation Agent
================================
A Streamlit-based chat agent that translates Chinese CSV columns
into multiple target languages using GPT-4o.

Design philosophy:
- Explicit agent stages (not a single megaprompt)
- Clear user confirmation before heavy operations
- Batch processing with progress feedback
- Graceful error handling and fallbacks
"""

import streamlit as st
import pandas as pd
import openai
import io
import json
import re
import time
from typing import Optional

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Translation Agent",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────
DEFAULTS = {
    "messages": [],
    "stage": "init",          # init → analyzing → confirming → translating → done
    "df": None,
    "filename": "",
    "chinese_cols": [],
    "selected_cols": [],
    "target_languages": [],
    "result_df": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def get_client() -> Optional[openai.OpenAI]:
    """
    Resolve API key and base_url.
    Supports both official OpenAI and OpenAI-compatible proxies (e.g. 云雾AI).
    Priority: sidebar input > Streamlit secrets.
    """
    key = st.session_state.get("api_key_input", "").strip()
    base_url = st.session_state.get("base_url_input", "").strip()

    if not key:
        try:
            key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass
    if not base_url:
        try:
            base_url = st.secrets.get("OPENAI_BASE_URL", "")
        except Exception:
            pass

    if not key:
        return None

    kwargs = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url  # e.g. https://yunwu.ai/v1

    return openai.OpenAI(**kwargs)


def is_chinese_column(series: pd.Series) -> bool:
    """Return True if ≥50% of non-null sampled values contain Chinese characters."""
    sample = series.dropna().astype(str).head(20)
    if len(sample) == 0:
        return False
    hits = sum(1 for t in sample if re.search(r"[一-鿿]", t))
    return hits / len(sample) >= 0.5


def translate_batch(client: openai.OpenAI, texts: list[str], target_lang: str) -> list[str]:
    """
    Translate a batch of strings to target_lang using GPT-4o.
    Returns original texts on failure (graceful degradation).
    """
    # Skip empty batch
    non_empty = [(i, t) for i, t in enumerate(texts) if t and str(t).strip()]
    if not non_empty:
        return texts

    indices, payloads = zip(*non_empty)

    prompt = f"""You are a professional translator for game and software content.
Translate each Chinese text to {target_lang}.
Preserve numbers, special characters, and placeholders (e.g. {{0}}, %s).
Return ONLY a JSON array of exactly {len(payloads)} translated strings in the same order.

Input: {json.dumps(list(payloads), ensure_ascii=False)}"""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Return only a valid JSON array of strings."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                timeout=60,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```(?:json)?\n?|\n?```", "", raw).strip()
            translated = json.loads(raw)

            if len(translated) != len(payloads):
                raise ValueError(f"Expected {len(payloads)} items, got {len(translated)}")

            # Reconstruct full list with translations in correct positions
            result = list(texts)
            for pos, orig_idx in enumerate(indices):
                result[orig_idx] = translated[pos]
            return result

        except Exception as e:
            if attempt == 2:
                st.warning(f"⚠️ 批次翻译失败，保留原文: {e}")
                return list(texts)
            time.sleep(1.5 * (attempt + 1))

    return list(texts)


def parse_translation_request(client: openai.OpenAI, user_msg: str, available_cols: list[str]) -> dict:
    """
    Use LLM to extract:
      - columns: which columns to translate
      - languages: target language names (in English)
      - is_clear: whether intent is unambiguous
    """
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""Extract translation instructions from the user message.
Available columns: {json.dumps(available_cols)}

Return JSON:
{{
  "columns": ["col1", "col2"],   // columns from available list; empty = translate all
  "languages": ["Japanese", "English"],  // target languages in English
  "is_clear": true  // false if you can't determine at least one language
}}""",
            },
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        timeout=15,
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"columns": [], "languages": [], "is_clear": False}


def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})


def lang_col_suffix(lang: str) -> str:
    """Map language name to a short column suffix."""
    mapping = {
        "japanese": "ja", "english": "en", "korean": "ko",
        "spanish": "es", "french": "fr", "german": "de",
        "portuguese": "pt", "thai": "th", "vietnamese": "vi",
        "indonesian": "id", "arabic": "ar", "russian": "ru",
    }
    return mapping.get(lang.lower(), lang[:3].lower())


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 设置")

    st.text_input(
        "API Key",
        type="password",
        placeholder="sk-...",
        key="api_key_input",
        help="OpenAI API Key，或兼容接口的 Key（如云雾AI）",
    )

    st.text_input(
        "API Base URL（可选）",
        placeholder="https://yunwu.ai/v1",
        key="base_url_input",
        help="留空 = 官方 OpenAI。使用云雾等代理时填写对应地址。",
    )

    st.divider()

    # File uploader lives here so chat area stays clean
    uploaded = st.file_uploader(
        "上传 CSV 文件",
        type=["csv"],
        key="csv_upload",
        help="包含中文列的 CSV 文件，支持 100+ 行",
    )

    st.divider()
    st.caption("**Translation Agent** · Vibe Coding Demo")
    st.caption("Powered by GPT-4o · Built with Streamlit")

    if st.session_state.stage != "init":
        if st.button("🔄 重新开始", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v if not isinstance(v, list) else []
            st.rerun()


# ─────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────
st.title("🌐 Multilingual Translation Agent")
st.caption("Chat-based CSV translation powered by GPT-4o")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── STAGE: init ────────────────────────────────
if st.session_state.stage == "init":
    welcome = """👋 **你好！我是多语言翻译 Agent。**

我可以把 CSV 文件中的中文内容批量翻译成多种目标语言，并输出包含所有翻译列的新 CSV 文件。

**使用步骤：**
1. 在左侧边栏填入 **OpenAI API Key**
2. 上传含中文列的 **CSV 文件**
3. 告诉我翻译目标（哪些列 + 哪些语言）
4. 等我处理完后**下载结果**

请先在左侧上传 CSV 文件 👈"""

    add_message("assistant", welcome)
    with st.chat_message("assistant"):
        st.markdown(welcome)
    st.session_state.stage = "waiting_csv"
    st.rerun()

# ── STAGE: waiting_csv ─────────────────────────
elif st.session_state.stage == "waiting_csv":
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.session_state.filename = uploaded.name

            # Identify Chinese columns
            ch_cols = [c for c in df.columns if is_chinese_column(df[c])]
            st.session_state.chinese_cols = ch_cols

            analysis = f"""📂 **已加载文件：** `{uploaded.name}`

| 指标 | 值 |
|---|---|
| 数据行数 | **{len(df):,} 行** |
| 总列数 | {len(df.columns)} 列 |
| 所有列 | {', '.join(f'`{c}`' for c in df.columns)} |
| 检测到的中文列 | {', '.join(f'`{c}`' for c in ch_cols) if ch_cols else '⚠️ 未自动检测到，请手动指定'} |

---
请告诉我：**需要翻译哪些列，翻译成什么语言？**

💡 示例：*"把 item_name 和 description 翻译成日语和英语"*
或直接说：*"翻译所有中文列到 Japanese、Korean、English"*"""

            add_message("assistant", analysis)
            st.session_state.stage = "confirming"
            st.rerun()

        except Exception as e:
            st.error(f"❌ 文件读取失败：{e}")

# ── STAGE: confirming ──────────────────────────
elif st.session_state.stage == "confirming":
    with st.expander("📊 数据预览", expanded=False):
        st.dataframe(st.session_state.df.head(5), use_container_width=True)

    user_input = st.chat_input("指定翻译列和目标语言...")

    if user_input:
        add_message("user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        client = get_client()
        if not client:
            msg = "❌ 请先在左侧边栏填入 **OpenAI API Key**，我才能处理翻译请求。"
            add_message("assistant", msg)
            with st.chat_message("assistant"):
                st.markdown(msg)
        else:
            available = st.session_state.chinese_cols or list(st.session_state.df.columns)
            with st.spinner("解析中..."):
                parsed = parse_translation_request(client, user_input, available)

            if not parsed.get("is_clear") or not parsed.get("languages"):
                clarify = """🤔 我需要再明确一下：

- **翻译哪些列**？（列名，或说"所有中文列"）
- **目标语言**是什么？（如：日语、英语、韩语）

示例：*"把 name 列翻译成 Japanese 和 Korean"*"""
                add_message("assistant", clarify)
                with st.chat_message("assistant"):
                    st.markdown(clarify)
            else:
                # Resolve columns
                cols = parsed.get("columns") or st.session_state.chinese_cols or list(st.session_state.df.columns)
                # Validate columns exist
                cols = [c for c in cols if c in st.session_state.df.columns]
                if not cols:
                    cols = st.session_state.chinese_cols or list(st.session_state.df.columns)

                langs = parsed.get("languages", [])
                st.session_state.selected_cols = cols
                st.session_state.target_languages = langs

                total_cells = len(cols) * len(langs) * len(st.session_state.df)
                output_cols = [f"`{c}_{lang_col_suffix(l)}`" for c in cols for l in langs]

                confirm = f"""✅ **翻译计划确认**

| 项目 | 详情 |
|---|---|
| 翻译列 | {', '.join(f'`{c}`' for c in cols)} |
| 目标语言 | {', '.join(langs)} |
| 数据行数 | {len(st.session_state.df):,} 行 |
| 新增输出列 | {', '.join(output_cols)} |
| 预计翻译单元 | ~{total_cells:,} 个文本单元 |

回复 **"确认"** 或 **"开始"** 启动翻译。"""

                add_message("assistant", confirm)
                with st.chat_message("assistant"):
                    st.markdown(confirm)

                st.session_state.stage = "ready"
                st.rerun()

# ── STAGE: ready (awaiting final confirmation) ──
elif st.session_state.stage == "ready":
    user_input = st.chat_input('输入"确认"或"开始"以启动翻译...')

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 开始翻译", type="primary", use_container_width=True):
            add_message("user", "确认，开始翻译")
            st.session_state.stage = "translating"
            st.rerun()
    with col2:
        if st.button("✏️ 修改设置", use_container_width=True):
            st.session_state.stage = "confirming"
            st.rerun()

    if user_input and any(kw in user_input for kw in ["确认", "开始", "ok", "yes", "go", "start"]):
        add_message("user", user_input)
        st.session_state.stage = "translating"
        st.rerun()
    elif user_input:
        add_message("user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)
        hint = '请输入 **"确认"** 开始翻译，或点击上方按钮。'
        add_message("assistant", hint)
        with st.chat_message("assistant"):
            st.markdown(hint)

# ── STAGE: translating ─────────────────────────
elif st.session_state.stage == "translating":
    client = get_client()
    if not client:
        st.error("❌ 需要 API Key，请在侧边栏填写后重试。")
    else:
        with st.chat_message("assistant"):
            st.markdown("⏳ **翻译进行中，请稍候...**")
            progress = st.progress(0.0, text="初始化...")
            status = st.empty()

            df = st.session_state.df.copy()
            cols = st.session_state.selected_cols
            langs = st.session_state.target_languages
            BATCH = 20

            total_batches = sum(
                (len(df) + BATCH - 1) // BATCH for _ in cols for _ in langs
            )
            done = 0
            errors = 0

            for col in cols:
                for lang in langs:
                    suffix = lang_col_suffix(lang)
                    new_col = f"{col}_{suffix}"
                    texts = df[col].fillna("").astype(str).tolist()
                    result = []

                    for i in range(0, len(texts), BATCH):
                        batch = texts[i : i + BATCH]
                        translated = translate_batch(client, batch, lang)
                        result.extend(translated)
                        done += 1
                        pct = done / total_batches
                        row_end = min(i + BATCH, len(texts))
                        progress.progress(pct, text=f"翻译 `{col}` → {lang} … {row_end}/{len(texts)} 行")
                        status.caption(f"进度：{done}/{total_batches} 批次 | 错误：{errors}")

                    df[new_col] = result

            st.session_state.result_df = df
            progress.progress(1.0, text="✅ 全部完成！")
            status.empty()

        done_msg = f"""🎉 **翻译完成！**

| | |
|---|---|
| 原始列数 | {len(st.session_state.df.columns)} |
| 新增翻译列 | {len(cols) * len(langs)} |
| 总列数 | {len(df.columns)} |
| 总行数 | {len(df):,} |

⬇️ 在下方下载完整结果 CSV"""

        add_message("assistant", done_msg)
        st.session_state.stage = "done"
        st.rerun()

# ── STAGE: done ────────────────────────────────
elif st.session_state.stage == "done":
    if st.session_state.result_df is not None:
        rdf = st.session_state.result_df

        with st.expander("📊 结果预览（前 5 行）", expanded=True):
            st.dataframe(rdf.head(5), use_container_width=True)

        # Prepare download
        buf = io.BytesIO()
        rdf.to_csv(buf, index=False, encoding="utf-8-sig")
        buf.seek(0)

        out_name = st.session_state.filename.replace(".csv", "_translated.csv")

        st.download_button(
            label="⬇️ 下载翻译结果 CSV",
            data=buf,
            file_name=out_name,
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

        st.caption("文件使用 UTF-8 BOM 编码，可直接用 Excel 打开显示中文。")

        user_input = st.chat_input("还有其他问题？或翻译新文件请重新上传...")
        if user_input:
            add_message("user", user_input)
            with st.chat_message("user"):
                st.markdown(user_input)
            follow_up = "如需翻译新文件，请点击左侧边栏的 **🔄 重新开始** 按钮，或直接上传新的 CSV 文件。"
            add_message("assistant", follow_up)
            with st.chat_message("assistant"):
                st.markdown(follow_up)
