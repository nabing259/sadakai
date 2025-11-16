import streamlit as st
import time

from rag import main_chain, retriever  

st.set_page_config(
    page_title="SadakAI – Road Safety Intervention GPT",
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="expanded"
)

# ---- CUSTOM CSS ----
st.markdown(
    """
    <style>
        body {
            background-color: #121212;
        }
        .main {
            background-color: #121212;
        }
        h1, h2, h3, p, label, .stTextInput label {
            color: #ffffff !important;
        }
        .title-style {
            font-size: 48px;
            font-weight: 700;
            color: #ffffff;
            text-align: center;
            margin-bottom: 0px;
        }
        .subtitle-style {
            font-size: 20px;
            color: #87c7ff;
            text-align: center;
            margin-top: -10px;
            margin-bottom: 30px;
        }
        .output-box {
            background-color: #1a1a1a;
            padding: 1px;
            border-radius: 12px;
            border: 1px solid #333333;
        }
        .blue-header {
            color: #4aa8ff !important;
            font-weight: 700;
        }
        .warning-triangle {
            color: #ff4444;
            font-size: 22px;
            margin-right: 8px;
        }
        .stTextArea textarea {
            background-color: #181818 !important;
            color: #ffffff !important;
            border-radius: 10px;
            border: 1px solid #2e2e2e;
        }
        .stButton>button {
            background-color: #4aa8ff;
            color: #ffffff;
            font-weight: 600;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1d7fcc;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- HEADER WITH LOGO ----

col1, col2, col3 = st.columns([1,8,1])
with col1:
    st.image("images/SadakAI-v2.png", width=80)

with col2:
    st.markdown('<div class="title-style">SadakAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-style">AI-powered Road Safety Intervention System</div>', unsafe_allow_html=True)
with col3:
    st.image("images/Robot-3d-icon.png", width=80)

# ---- INPUT AREA ----

st.markdown("### 🧭 Describe your road safety issue:")
st.markdown('<div >', unsafe_allow_html=True)
user_query = st.text_area("", placeholder="Describe the road section, crash pattern, visibility issues, traffic behavior…", height=120)
st.markdown('</div>', unsafe_allow_html=True)

# ---- PROCESSING ----

if st.button("🔍 Analyze with SadakAI"):
    if not user_query.strip():
        st.error("Please enter a valid road safety description.")
    else:
        with st.spinner("🤖 SadakAI is Thinking Best Possible Interventions…", show_time=True):
            response = main_chain.invoke(user_query)

        # OUTPUT

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### <span class='blue-header'>📘 Recommended Interventions</span>", unsafe_allow_html=True)
        st.markdown('<div class="output-box">', unsafe_allow_html=True)
        st.write(response)
        st.markdown('</div>', unsafe_allow_html=True)

        # RETRIEVED CONTEXT

        with st.expander("📂 Show Retrieved Knowledge Chunks (IRC/WHO)"):
            docs = retriever.invoke(user_query)
            for i, d in enumerate(docs, 1):
                st.markdown(f"### Document {i}")
                st.write(d.page_content)

# FOOTER
st.markdown("<br><br><center style='color:#777;'>SadakAI © 2025 • Created by Garai Bros</center>", unsafe_allow_html=True)
