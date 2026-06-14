"""
app.py — Obesity Health Category Classifier
AIB02 UAS | Josua Nathanael Dharmawan | 38250005
UI: Dark minimalis, rapi, ramah orang awam
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cek Kategori Berat Badan",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS — DARK MINIMALIS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg:          #0f1117;
    --bg-panel:    #161b27;
    --card:        #1c2333;
    --border:      rgba(255,255,255,0.08);
    --border-med:  rgba(255,255,255,0.13);
    --text:        #f1f5f9;
    --text-sub:    #94a3b8;
    --text-muted:  #475569;
    --accent:      #3b82f6;
    --accent-dim:  rgba(59,130,246,0.15);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--text);
}

.stApp {
    background: var(--bg);
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 2.5rem 1.5rem 4rem 1.5rem !important;
    max-width: 760px !important;
}

/* ── HEADER ── */
.app-header {
    text-align: center;
    padding: 2rem 0 1.75rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.app-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}
.app-header p {
    font-size: 0.875rem;
    color: var(--text-sub);
    margin: 0;
    line-height: 1.5;
}
.app-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid rgba(59,130,246,0.3);
    margin-top: 0.8rem;
    letter-spacing: 0.02em;
}

/* ── SECTION LABEL ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.75rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── CARD ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

/* ── BMI STRIP ── */
.bmi-strip {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 1.25rem 0;
}
.bmi-number {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
}
.bmi-info-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.bmi-info-cat {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text);
    margin-top: 0.15rem;
}

/* ── BUTTON ── */
.stButton > button {
    width: 100% !important;
    background: var(--accent) !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    cursor: pointer !important;
    transition: opacity 0.15s !important;
    margin-top: 0.75rem !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* ── RESULT CARD ── */
.result-card {
    border-radius: 10px;
    padding: 1.5rem 1.75rem;
    margin: 1.5rem 0;
    border: 1px solid;
}
.result-tag {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    opacity: 0.7;
}
.result-title {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.result-advice {
    font-size: 0.82rem;
    margin-top: 0.5rem;
    opacity: 0.75;
    line-height: 1.5;
}

/* ── PROB BARS ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.55rem;
    font-size: 0.775rem;
}
.prob-name {
    width: 175px;
    color: var(--text-sub);
    flex-shrink: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.prob-bar-bg {
    flex: 1;
    height: 5px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 3px;
}
.prob-pct {
    width: 40px;
    text-align: right;
    color: var(--text-sub);
    font-weight: 500;
    font-size: 0.72rem;
}

/* ── FORM ELEMENTS ── */
label[data-testid="stWidgetLabel"] p {
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    color: var(--text) !important;
}

div[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border-med) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 0.85rem !important;
}

div[data-testid="stNumberInput"] input {
    background: var(--card) !important;
    border: 1px solid var(--border-med) !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    text-align: center !important;
    color: var(--text) !important;
}

.stSlider [data-testid="stThumbValue"] {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    background: var(--card) !important;
    border: 1px solid var(--border-med) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
}

[data-testid="stAlert"] {
    background: rgba(59,130,246,0.08) !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    color: var(--text-sub) !important;
}

/* ── TABS ── */
div[data-baseweb="tab-list"] {
    background: var(--card) !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 3px !important;
    border: 1px solid var(--border) !important;
}
div[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--text-sub) !important;
    background: transparent !important;
    border-radius: 6px !important;
    padding: 0.45rem 1.2rem !important;
    border: none !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    transition: all 0.15s !important;
}
div[aria-selected="true"] {
    color: var(--text) !important;
    background: var(--bg-panel) !important;
    font-weight: 600 !important;
}
div[data-baseweb="tab-highlight"],
div[data-baseweb="tab-border"] { display: none !important; }

/* ── FOOTER ── */
.app-footer {
    text-align: center;
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS & METADATA
# ─────────────────────────────────────────────
CLASS_META = {
    "Insufficient_Weight": {
        "label": "Berat Badan Kurang",
        "color": "#60a5fa", "bg": "rgba(59,130,246,0.1)", "border": "rgba(59,130,246,0.25)",
        "emoji": "📉",
        "advice": "Tingkatkan asupan makanan bergizi. Konsultasikan dengan dokter atau ahli gizi."
    },
    "Normal_Weight": {
        "label": "Berat Badan Normal",
        "color": "#4ade80", "bg": "rgba(34,197,94,0.1)", "border": "rgba(34,197,94,0.25)",
        "emoji": "✅",
        "advice": "Pertahankan pola makan seimbang dan tetap aktif bergerak."
    },
    "Overweight_Level_I": {
        "label": "Kelebihan Berat Ringan",
        "color": "#fbbf24", "bg": "rgba(245,158,11,0.1)", "border": "rgba(245,158,11,0.25)",
        "emoji": "⚠️",
        "advice": "Kurangi makanan berlemak dan perbanyak aktivitas fisik."
    },
    "Overweight_Level_II": {
        "label": "Kelebihan Berat Sedang",
        "color": "#fb923c", "bg": "rgba(249,115,22,0.1)", "border": "rgba(249,115,22,0.25)",
        "emoji": "⚠️",
        "advice": "Pertimbangkan program diet terstruktur. Dianjurkan konsultasi dokter."
    },
    "Obesity_Type_I": {
        "label": "Obesitas Tingkat 1",
        "color": "#f87171", "bg": "rgba(239,68,68,0.1)", "border": "rgba(239,68,68,0.25)",
        "emoji": "🔴",
        "advice": "Sangat disarankan konsultasi medis dan program penurunan berat badan."
    },
    "Obesity_Type_II": {
        "label": "Obesitas Tingkat 2",
        "color": "#ef4444", "bg": "rgba(220,38,38,0.12)", "border": "rgba(220,38,38,0.3)",
        "emoji": "🔴",
        "advice": "Diperlukan penanganan medis. Segera hubungi dokter."
    },
    "Obesity_Type_III": {
        "label": "Obesitas Tingkat 3",
        "color": "#fca5a5", "bg": "rgba(185,28,28,0.15)", "border": "rgba(185,28,28,0.4)",
        "emoji": "🚨",
        "advice": "Kondisi serius — diperlukan penanganan medis segera."
    },
}

@st.cache_resource
def load_model():
    base = "model"
    if not os.path.exists(f"{base}/mlp_obesity.pkl"):
        return None, None, None, None, None
    model     = joblib.load(f"{base}/mlp_obesity.pkl")
    scaler    = joblib.load(f"{base}/scaler.pkl")
    le_tgt    = joblib.load(f"{base}/label_encoder_target.pkl")
    le_feat   = joblib.load(f"{base}/label_encoders_features.pkl")
    feat_cols = joblib.load(f"{base}/feature_cols.pkl")
    return model, scaler, le_tgt, le_feat, feat_cols

def bmi_category(bmi):
    if bmi < 18.5: return "Kurang"
    if bmi < 25.0: return "Normal"
    if bmi < 30.0: return "Berlebih"
    return "Obesitas"

def bmi_color(bmi):
    if bmi < 18.5: return "#60a5fa"
    if bmi < 25.0: return "#4ade80"
    if bmi < 30.0: return "#fbbf24"
    return "#f87171"

# Konversi sayuran ke angka (sesuai dataset: 1=jarang, 2=kadang, 3=sering)
SAYURAN_MAP = {
    "Jarang": 1.0,
    "Kadang-kadang": 2.0,
    "Sering": 3.0,
}

def prob_bar_html(name, pct, color, is_top):
    label = CLASS_META.get(name, {}).get("label", name)
    bar_color = color if is_top else "rgba(255,255,255,0.08)"
    text_color = "#f1f5f9" if is_top else "#475569"
    return f"""
    <div class="prob-row">
        <div class="prob-name" style="color:{text_color};font-weight:{'600' if is_top else '400'}">{label}</div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{pct*100:.1f}%;background:{bar_color};"></div>
        </div>
        <div class="prob-pct" style="color:{text_color}">{pct*100:.1f}%</div>
    </div>"""

def run_prediction(raw, feat_cols, le_feat, le_tgt, scaler, model):
    input_df = pd.DataFrame([raw])[feat_cols]
    cat_cols_model = input_df.select_dtypes(include=["object", "str"]).columns
    for col in cat_cols_model:
        if col in le_feat:
            le = le_feat[col]
            val = input_df[col].astype(str).values[0]
            input_df[col] = le.transform([val]) if val in le.classes_ else 0
    X_input  = scaler.transform(input_df.values.reshape(1, -1))
    pred_idx = model.predict(X_input)[0]
    pred_cls = le_tgt.classes_[pred_idx]
    proba    = model.predict_proba(X_input)[0]
    return pred_cls, proba, X_input

def show_result(pred_cls, proba, le_tgt):
    meta = CLASS_META.get(pred_cls, {
        "label": pred_cls, "color": "#60a5fa",
        "bg": "rgba(59,130,246,0.1)", "border": "rgba(59,130,246,0.25)",
        "emoji": "📊", "advice": ""
    })
    st.markdown(f"""
    <div class="result-card" style="background:{meta['bg']};border-color:{meta['border']};">
        <div class="result-tag" style="color:{meta['color']}">Hasil Prediksi</div>
        <div class="result-title" style="color:{meta['color']}">{meta['emoji']} {meta['label']}</div>
        <div class="result-advice" style="color:{meta['color']}">{meta['advice']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Distribusi Probabilitas</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    sorted_idx = np.argsort(proba)[::-1]
    bars = ""
    for i, idx in enumerate(sorted_idx):
        cls = le_tgt.classes_[idx]
        c = CLASS_META.get(cls, {}).get("color", "#e5e7eb")
        bars += prob_bar_html(cls, proba[idx], c, i == 0)
    st.markdown(bars, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model, scaler, le_tgt, le_feat, feat_cols = load_model()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🩺 Cek Kategori Berat Badan</h1>
    <p>Isi data di bawah untuk mengetahui kategori berat badanmu berdasarkan gaya hidup</p>
    <span class="app-badge">Akurasi Model 94.09% · ANN/MLP</span>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.warning("⚠️ Model belum ditemukan. Jalankan script training dulu: `python Project_UAS_Josua_Nathanael_Dharmawan_38250005.py`")
    st.stop()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_manual, tab_cv = st.tabs(["✏️  Input Manual", "📷  Foto Wajah (CV)"])


# ═══════════════════════════════════════════
# TAB 1 — MANUAL INPUT
# ═══════════════════════════════════════════
with tab_manual:

    # ── BAGIAN 1: Data Diri ──
    st.markdown('<div class="section-label">Data Diri</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"],
                              format_func=lambda x: "Laki-laki" if x == "Male" else "Perempuan")
        age    = st.slider("Usia (tahun)", 10, 80, 25)
    with c2:
        height = st.number_input("Tinggi Badan (m)", 1.40, 2.20, 1.70, 0.01, format="%.2f")
        weight = st.number_input("Berat Badan (kg)", 30.0, 200.0, 70.0, 0.5, format="%.1f")

    fam_hist = st.selectbox(
        "Ada anggota keluarga yang pernah kelebihan berat badan?",
        ["yes", "no"],
        format_func=lambda x: "Ya" if x == "yes" else "Tidak"
    )

    # BMI Live
    bmi = weight / (height ** 2)
    bmi_cat = bmi_category(bmi)
    bc = bmi_color(bmi)
    st.markdown(f"""
    <div class="bmi-strip">
        <div class="bmi-number" style="color:{bc}">{bmi:.1f}</div>
        <div>
            <div class="bmi-info-label">BMI kamu saat ini</div>
            <div class="bmi-info-cat" style="color:{bc}">{bmi_cat}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── BAGIAN 2: Kebiasaan Makan ──
    st.markdown('<div class="section-label">Kebiasaan Makan</div>', unsafe_allow_html=True)

    favc_raw = st.selectbox(
        "Seberapa sering makan makanan tinggi kalori? (gorengan, fastfood, makanan berminyak)",
        ["yes", "no"],
        format_func=lambda x: "Sering / hampir tiap hari" if x == "yes" else "Jarang / sesekali saja"
    )

    sayuran_pilihan = st.select_slider(
        "Seberapa sering makan sayur atau buah?",
        options=["Jarang", "Kadang-kadang", "Sering"],
        value="Kadang-kadang"
    )
    fcvc = SAYURAN_MAP[sayuran_pilihan]

    ncp_raw = st.select_slider(
        "Berapa kali makan besar dalam sehari? (sarapan, makan siang, makan malam)",
        options=[1, 2, 3],
        value=3
    )
    ncp = float(ncp_raw)

    caec = st.selectbox(
        "Suka makan camilan di luar jam makan?",
        ["no", "Sometimes", "Frequently", "Always"],
        format_func=lambda x: {
            "no": "Tidak pernah",
            "Sometimes": "Kadang-kadang",
            "Frequently": "Sering",
            "Always": "Selalu / hampir tiap saat"
        }.get(x, x)
    )

    calc = st.selectbox(
        "Seberapa sering minum minuman beralkohol?",
        ["no", "Sometimes", "Frequently", "Always"],
        format_func=lambda x: {
            "no": "Tidak pernah",
            "Sometimes": "Kadang-kadang",
            "Frequently": "Sering",
            "Always": "Selalu"
        }.get(x, x)
    )

    # ── BAGIAN 3: Kebiasaan Hidup ──
    st.markdown('<div class="section-label">Kebiasaan Hidup</div>', unsafe_allow_html=True)

    ch2o_raw = st.slider(
        "Berapa liter air putih yang kamu minum per hari?",
        1.0, 7.0, 2.0, 0.5,
        format="%.1f liter"
    )
    ch2o = ch2o_raw

    faf_raw = st.slider(
        "Berapa kali kamu olahraga dalam seminggu?",
        0, 7, 3,
        format="%d kali"
    )
    faf = min(float(faf_raw) / 7.0 * 3.0, 3.0)  # normalisasi ke skala 0–3 sesuai dataset

    tue_raw = st.slider(
        "Berapa jam per hari kamu pakai HP / komputer / nonton layar?",
        0, 16, 4,
        format="%d jam"
    )
    # Konversi jam ke skala 0–2 sesuai dataset: 0=<1h, 1=1-5h, 2=>5h
    if tue_raw < 1:
        tue = 0.0
    elif tue_raw <= 5:
        tue = 1.0
    else:
        tue = 2.0

    smoke = st.selectbox(
        "Apakah kamu merokok?",
        ["no", "yes"],
        format_func=lambda x: "Tidak" if x == "no" else "Ya"
    )

    scc = st.selectbox(
        "Apakah kamu menghitung atau memantau kalori yang kamu makan?",
        ["no", "yes"],
        format_func=lambda x: "Tidak" if x == "no" else "Ya"
    )

    mtrans = st.selectbox(
        "Transportasi yang biasa kamu gunakan sehari-hari?",
        ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"],
        format_func=lambda x: {
            "Automobile": "Mobil",
            "Motorbike": "Motor",
            "Bike": "Sepeda",
            "Public_Transportation": "Transportasi Umum (KRL/MRT/Bus)",
            "Walking": "Jalan Kaki"
        }.get(x, x)
    )

    # ── PREDICT ──
    predict_btn = st.button("🔍  Cek Kategori Berat Badan", key="predict_manual")

    if predict_btn:
        raw = {
            "Gender": gender,
            "Age": float(age),
            "Height": float(height),
            "Weight": float(weight),
            "family_history_with_overweight": fam_hist,
            "FAVC": favc_raw,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": float(ch2o),
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans,
        }
        pred_cls, proba, _ = run_prediction(raw, feat_cols, le_feat, le_tgt, scaler, model)
        show_result(pred_cls, proba, le_tgt)

    st.caption("⚠️ Hasil ini bukan diagnosis medis. Konsultasikan ke dokter untuk pemeriksaan lebih lanjut.")


# ═══════════════════════════════════════════
# TAB 2 — FACE SCAN (CV)
# ═══════════════════════════════════════════
with tab_cv:
    st.markdown('<div class="section-label">Analisis Wajah via Kamera</div>', unsafe_allow_html=True)

    st.info(
        "Sistem akan mendeteksi proporsi wajah menggunakan OpenCV untuk memperkirakan "
        "kategori berat badan. Foto harus frontal, pencahayaan cukup, tanpa masker."
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        cv_gender = st.selectbox("Jenis Kelamin", ["Male", "Female"], key="cv_g",
                                 format_func=lambda x: "Laki-laki" if x == "Male" else "Perempuan")
    with c2:
        cv_age = st.slider("Usia", 10, 80, 25, key="cv_a")
    with c3:
        cv_height = st.number_input("Tinggi (m)", 1.40, 2.20, 1.70, 0.01, format="%.2f", key="cv_h")

    use_weight = st.toggle("Tambahkan berat badan (meningkatkan akurasi)", value=False)
    if use_weight:
        cv_weight = st.number_input("Berat Badan (kg)", 30.0, 200.0, 70.0, 0.5, format="%.1f", key="cv_w")
    else:
        cv_weight = None

    st.markdown('<div class="section-label">Upload Foto</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Pilih foto JPG / PNG", type=["jpg", "jpeg", "png"])

    if uploaded:
        try:
            import cv2
            img_pil = Image.open(uploaded).convert("RGB")
            img_np  = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            hc_path      = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            hc_eye       = cv2.data.haarcascades + "haarcascade_eye.xml"
            face_cascade = cv2.CascadeClassifier(hc_path)
            eye_cascade  = cv2.CascadeClassifier(hc_eye)
            faces        = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

            col_img, col_res = st.columns([1, 1], gap="large")

            with col_img:
                if len(faces) == 0:
                    st.error("Wajah tidak terdeteksi. Coba foto yang lebih terang atau lebih frontal.")
                else:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    vis = img_bgr.copy()
                    # Kotak wajah — warna biru bersih
                    cv2.rectangle(vis, (x, y), (x+w, y+h), (37, 99, 235), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.ellipse(vis, (x+ex+ew//2, y+ey+eh//2), (ew//2, eh//2), 0, 0, 360, (22, 163, 74), 1)
                    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    st.image(vis_rgb, use_container_width=True)

                    # Hitung fitur wajah
                    far = w / h
                    if len(eyes) >= 2:
                        e1, e2 = sorted(eyes[:2], key=lambda e: e[0])
                        cx1 = x + e1[0] + e1[2]//2
                        cx2 = x + e2[0] + e2[2]//2
                        eye_span  = abs(cx2 - cx1) / w
                        avg_eye_w = (e1[2] + e2[2]) / 2.0 / w
                    else:
                        eye_span  = 0.42
                        avg_eye_w = 0.13

                    adiposity_score = max(0, min(100,
                        (far - 0.65) / 0.35 * 50 +
                        (0.5 - eye_span) / 0.5 * 30 +
                        (0.18 - avg_eye_w) / 0.18 * 20
                    ))
                    bmi_cv = 17.0 + adiposity_score * 0.28

                    if cv_weight is not None:
                        bmi_actual = cv_weight / (cv_height ** 2)
                        bmi_final  = bmi_actual * 0.6 + bmi_cv * 0.4
                    else:
                        bmi_final = bmi_cv

            with col_res:
                if len(faces) > 0:
                    st.markdown('<div class="section-label">Metrik Wajah</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="card">
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;">
                            <div>
                                <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">Rasio Wajah</div>
                                <div style="font-size:1.3rem;font-weight:700;color:#f1f5f9;">{far:.3f}</div>
                            </div>
                            <div>
                                <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">Jarak Mata</div>
                                <div style="font-size:1.3rem;font-weight:700;color:#f1f5f9;">{eye_span:.3f}</div>
                            </div>
                            <div>
                                <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">Lebar Mata</div>
                                <div style="font-size:1.3rem;font-weight:700;color:#f1f5f9;">{avg_eye_w:.3f}</div>
                            </div>
                            <div>
                                <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">Skor Lemak Wajah</div>
                                <div style="font-size:1.3rem;font-weight:700;color:{'#f87171' if adiposity_score>60 else '#4ade80' if adiposity_score<30 else '#fbbf24'};">{adiposity_score:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Prediksi dari CV
                    raw_cv = {
                        "Gender": cv_gender, "Age": float(cv_age), "Height": float(cv_height),
                        "Weight": float(bmi_final * cv_height**2),
                        "family_history_with_overweight": "yes" if adiposity_score > 60 else "no",
                        "FAVC": "yes", "FCVC": 2.0, "NCP": 3.0,
                        "CAEC": "Sometimes", "SMOKE": "no", "CH2O": 2.0,
                        "SCC": "no", "FAF": 1.0, "TUE": 1.0,
                        "CALC": "no", "MTRANS": "Public_Transportation",
                    }
                    pred_cls, proba, _ = run_prediction(raw_cv, feat_cols, le_feat, le_tgt, scaler, model)
                    show_result(pred_cls, proba, le_tgt)
                    st.caption(f"Estimasi BMI dari wajah: {bmi_final:.1f}")
                    st.caption("⚠️ Mode foto memberikan estimasi kasar. Gunakan Input Manual untuk hasil lebih akurat.")

        except Exception as e:
            st.error(f"Error memproses gambar: {e}")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Josua Nathanael Dharmawan · 38250005 · AIB02 UAS<br>
    MLP Classifier (sklearn) · OpenCV Haarcascade · UCI Obesity Dataset · Akurasi 94.09%
</div>
""", unsafe_allow_html=True)