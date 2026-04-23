import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import re

# ================= PAGE =================
st.set_page_config(page_title="Emotion AI Dashboard", layout="wide")

# ================= MODEL =================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ================= SESSION =================
if "chat" not in st.session_state:
    st.session_state.chat = []

if "selected_text" not in st.session_state:
    st.session_state.selected_text = None

# ================= LABELS =================
label_map = {
    0: "Sad",
    1: "Happy",
    2: "Love",
    3: "Angry",
    4: "Fear",
    5: "Surprise"
}

colors = ["#60a5fa", "#facc15", "#fb7185", "#ef4444", "#a78bfa", "#34d399"]

# ================= STYLE =================
st.markdown("""
<style>
.stApp { background-color: #0b1220; }

/* TEXT */
h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; }
p, label { color: #ffffff !important; }

/* INPUT */
textarea {
    background-color: white !important;
    color: black !important;
    border-radius: 8px !important;
}

/* BUTTON */
.stButton > button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
}

/* RESULT */
.result {
    background: #111827;
    padding: 12px;
    margin-top: 10px;
    border-radius: 8px;
    border-left: 4px solid #60a5fa;
    color: white;
}

/* NOTE */
.note {
    background: #111827;
    padding: 10px;
    margin-top: 15px;
    border-radius: 8px;
    color: #cbd5e1;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* SIDEBAR BUTTONS */
section[data-testid="stSidebar"] .stButton > button {
    background-color: #1f2937 !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #374151 !important;
}

/* DOWNLOAD BUTTON */
section[data-testid="stSidebar"] .stDownloadButton > button {
    background-color: #22c55e !important;
    color: white !important;
    font-weight: 600;
    border-radius: 8px;
}
section[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background-color: #16a34a !important;
}

/* ===== SPACING FIX ===== */
section[data-testid="stSidebar"] .block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0.5rem !important;
}

section[data-testid="stSidebar"] .stButton,
section[data-testid="stSidebar"] .stDownloadButton {
    margin-top: 0.2rem !important;
    margin-bottom: 0.2rem !important;
}

section[data-testid="stSidebar"] hr {
    margin-top: 0.4rem !important;
    margin-bottom: 0.4rem !important;
}

/* plotly title fix */
.js-plotly-plot .plotly .gtitle {
    fill: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("🧠 Emotion Intelligence Report")

# ================= INPUT =================
user_input = st.text_area("Enter your text here", height=120)

# ================= VALIDATION =================
def is_valid_text(text):
    if len(text.split()) < 2:
        return False
    if re.fullmatch(r"[a-zA-Z]{10,}", text) and len(set(text)) < 5:
        return False
    return True

# ================= SUGGESTION =================
def get_suggestion(emotion):
    return {
        "Happy": "Keep smiling 😊 and spread positivity ✨",
        "Sad": "It's okay 🌿 Take rest and care for yourself.",
        "Angry": "Pause 🌊 breathe before reacting.",
        "Fear": "Stay calm 🤍 You are strong.",
        "Love": "Beautiful 💖 Express kindness.",
        "Surprise": "Embrace it ✨ Life is full of moments."
    }.get(emotion, "Stay balanced 🌱 keep moving forward.")

# ================= ANALYSIS =================
def analyze(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    proba = model.predict_proba(vec)

    emotion = label_map[int(pred[0])]
    confidence = np.max(proba[0]) * 100

    return emotion, confidence, proba

def show_result(text):
    emotion, confidence, proba = analyze(text)

    df = pd.DataFrame({
        "emotion": [label_map[int(c)] for c in model.classes_],
        "value": [p * 100 for p in proba[0]]
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["emotion"],
        y=df["value"],
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in df["value"]],
        textposition="outside"
    ))

    fig.update_layout(
        title=dict(text="Emotion Intelligence Report", font=dict(color="white", size=22)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="result">
        <h3>{emotion}</h3>
        <p>Confidence: {confidence:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    suggestion = get_suggestion(emotion)
    st.markdown(f"""
    <div class="note">
💡 <b>Suggestion:</b> {suggestion}
    </div>
    """, unsafe_allow_html=True)

# ================= ANALYZE BUTTON =================
if st.button("Analyze Emotion"):
    if not user_input.strip():
        st.warning("Please enter text")
        st.stop()

    if not is_valid_text(user_input):
        st.error("⚠️ Please enter meaningful text")
        st.stop()

    emotion, confidence, _ = analyze(user_input)

    st.session_state.selected_text = user_input
    st.session_state.chat.append({
        "text": user_input,
        "emotion": emotion,
        "confidence": confidence
    })

# ================= SHOW RESULT =================
if st.session_state.selected_text:
    show_result(st.session_state.selected_text)

# ================= SIDEBAR =================
st.sidebar.title("💬 Chat History")

# DOWNLOAD TOP
if st.session_state.chat:
    st.sidebar.markdown("### 📥 Export")

    df_download = pd.DataFrame(st.session_state.chat)

    st.sidebar.download_button(
        "⬇️ Download Report",
        df_download.to_csv(index=False),
        "emotion_history.csv",
        "text/csv"
    )

st.sidebar.markdown("---")

# CLEAR
if st.sidebar.button("🧹 Clear History"):
    st.session_state.chat = []
    st.session_state.selected_text = None

st.sidebar.markdown("---")

# CHAT LIST
for i, item in enumerate(st.session_state.chat[::-1]):
    col1, col2 = st.sidebar.columns([4,1])

    if col1.button(item["text"][:25], key=f"select_{i}"):
        st.session_state.selected_text = item["text"]

    if col2.button("❌", key=f"del_{i}"):
        st.session_state.chat.pop(len(st.session_state.chat)-1-i)
        st.rerun()

# ================= NOTE =================
st.markdown("""
<div class="note">
⚠️ <b>Note:</b> This AI is based on machine learning and may not always be fully accurate.
</div>
""", unsafe_allow_html=True)