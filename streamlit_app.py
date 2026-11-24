import hashlib
import io
import json
import os

import streamlit as st
from groq import Groq
from gtts import gTTS

st.set_page_config(page_title="Nói chuyện bằng 2 ngôn ngữ - Groq AI", page_icon="🗣️", layout="centered")

# Giữ 2 menu ngôn ngữ trên cùng 1 dòng, kể cả trên iPhone (Streamlit mặc định xếp chồng cột trên màn hình hẹp)
st.markdown(
    """
    <style>
    div[data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 0.5rem !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        flex: 1 1 0 !important;
        width: auto !important;
        min-width: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 'dk' trên giao diện = tiếng Đan Mạch (mã chuẩn 'da')
LANGS = {
    "vi": {"code": "vi", "tts": "vi", "name": "Vietnamese"},
    "en": {"code": "en", "tts": "en", "name": "English"},
    "fr": {"code": "fr", "tts": "fr", "name": "French"},
    "zh": {"code": "zh", "tts": "zh-CN", "name": "Chinese"},
    "dk": {"code": "da", "tts": "da", "name": "Danish"},
}
BY_CODE = {v["code"]: k for k, v in LANGS.items()}  # code -> nhãn


def get_secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


API_KEY = get_secret("GROQ_API_KEY")
STT_MODEL = get_secret("STT_MODEL", "whisper-large-v3-turbo")
LLM_MODEL = get_secret("LLM_MODEL", "qwen/qwen3.8-27b")


# ---------------- Gọi Groq ----------------
def transcribe(client, audio_bytes, lang_code=None):
    kwargs = dict(
        file=("speech.wav", audio_bytes),
        model=STT_MODEL,
        response_format="json",
        temperature=0,
    )
    if lang_code:  # 2 menu giống nhau -> biết chắc ngôn ngữ
        kwargs["language"] = lang_code
    res = client.audio.transcriptions.create(**kwargs)
    return (res.text or "").strip()


def respond(client, a, b, text, history):
    A, B = LANGS[a], LANGS[b]
    if a == b:
        system = (
            f"You are a friendly voice assistant. The user speaks {A['name']}. "
            f"Always answer in {A['name']}, briefly and naturally (1-3 short sentences), "
            "as your answer will be read aloud: no markdown, no lists, no emojis. "
            f'Reply ONLY with JSON: {{"source":"{A["code"]}","output":"<your answer>"}}'
        )
        messages = [{"role": "system", "content": system}] + history[-10:] + [{"role": "user", "content": text}]
        temperature = 0.6
    else:
        system = (
            f"You are a live interpreter between {A['name']} (code \"{A['code']}\") and {B['name']} (code \"{B['code']}\"). "
            f"The user message is a speech transcript (may contain recognition errors) in EITHER {A['name']} OR {B['name']}. "
            "1) Decide which of the two languages it is written in. "
            "2) Translate it faithfully and naturally into the OTHER language. Do not answer it, do not explain. "
            'If the text is not clearly in either language (noise, silence artifacts), set source to "none" and output to "". '
            f'Reply ONLY with JSON: {{"source":"<{A["code"]}|{B["code"]}|none>","output":"<translation>"}}'
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
        temperature = 0.1

    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    data = json.loads(res.choices[0].message.content)
    source = data.get("source")
    output = (data.get("output") or "").strip()
    if a == b:
        target = a_code = A["code"]
    elif source == A["code"]:
        target = B["code"]
    elif source == B["code"]:
        target = A["code"]
    else:
        target = None
    return source, target, output


def speak_bytes(text, label):
    buf = io.BytesIO()
    gTTS(text=text, lang=LANGS[label]["tts"]).write_to_fp(buf)
    return buf.getvalue()


# ---------------- Giao diện ----------------
st.header("**Chat in multiple languages**")

if not API_KEY:
    st.error("Chưa có GROQ_API_KEY. Hãy đặt trong .streamlit/secrets.toml hoặc biến môi trường.")
    st.stop()

client = Groq(api_key=API_KEY)

ss = st.session_state
ss.setdefault("running", False)
ss.setdefault("rec_key", 0)
ss.setdefault("last_hash", None)
ss.setdefault("log", [])        # danh sách tin nhắn hiển thị
ss.setdefault("llm_history", [])  # lịch sử chat với AI (khi 2 menu giống nhau)

c1, c2 = st.columns(2)
l1 = c1.selectbox("Lang 1", list(LANGS), index=0, key="l1", disabled=ss.running)
l2 = c2.selectbox("Lang 2", list(LANGS), index=1, key="l2", disabled=ss.running)

if st.button("Stop" if ss.running else "Start", type="primary", use_container_width=True):
    ss.running = not ss.running
    ss.rec_key += 1          # làm mới ô ghi âm
    ss.last_hash = None
    if ss.running:
        ss.log = []
        ss.llm_history = []
    st.rerun()

if ss.running:
    a, b = LANGS[l1]["code"], LANGS[l2]["code"]
    st.caption(
        "Chế độ chat với AI: hãy đặt câu hỏi." if a == b
        else f"Chế độ phiên dịch {l1} ⇄ {l2}: mỗi người nhấn micro, nói, rồi nhấn lại để dừng."
    )
    audio = st.audio_input("Nhấn micro để nói, nhấn lại để dừng", key=f"rec_{ss.rec_key}")

    if audio is not None:
        data = audio.getvalue()
        h = hashlib.md5(data).hexdigest()
        if h != ss.last_hash:  # chỉ xử lý mỗi đoạn ghi âm một lần
            ss.last_hash = h
            try:
                with st.spinner("Đang nhận dạng..."):
                    text = transcribe(client, data, a if a == b else None)
                if len(text) < 2:
                    st.warning("Không nghe rõ, hãy nói lại.")
                else:
                    with st.spinner("Đang dịch / trả lời..."):
                        source, target, output = respond(client, a, b, text, ss.llm_history)
                    if source == "none" or not output or not target:
                        st.warning("Không nhận ra ngôn ngữ, hãy nói lại.")
                    else:
                        if a == b:
                            ss.llm_history += [
                                {"role": "user", "content": text},
                                {"role": "assistant", "content": json.dumps({"source": a, "output": output}, ensure_ascii=False)},
                            ]
                            ss.llm_history = ss.llm_history[-20:]
                        with st.spinner("Đang tạo giọng nói..."):
                            mp3 = speak_bytes(output, BY_CODE[target])
                        ss.log.insert(0, {
                            "src": BY_CODE.get(source, "?"), "orig": text,
                            "tgt": BY_CODE[target], "out": output,
                            "audio": mp3, "new": True,
                        })
            except Exception as e:
                st.error(f"Lỗi: {e}")

# Hiển thị hội thoại (mới nhất ở trên); chỉ tự phát âm thanh của tin mới nhất
for item in ss.log:
    with st.container(border=True):
        st.caption(f"[{item['src']}] {item['orig']}")
        st.markdown(f"**[{item['tgt']}]** {item['out']}")
        st.audio(item["audio"], format="audio/mp3", autoplay=item["new"])
    item["new"] = False

