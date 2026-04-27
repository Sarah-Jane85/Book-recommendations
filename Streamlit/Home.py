# ── Home.py — The Other Shelf landing page ──────────────────────────────────
import streamlit as st
import streamlit.components.v1 as components
import base64
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from Components.shared import set_page_style

st.set_page_config(
    page_title="The Other Shelf",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

set_page_style()

# ── Load images as base64 ────────────────────────────────────────────────────
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

ASSETS      = Path(__file__).resolve().parent / "Assets"
img_fantasy = img_to_base64(ASSETS / "book_fantasy.png")
img_graphic = img_to_base64(ASSETS / "book_graphic.png")
img_nonfic  = img_to_base64(ASSETS / "book_nonfiction.png")

# ── Title ────────────────────────────────────────────────────────────────────
st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem 0;'>
        <h1 style='font-size:4.5rem; color:#F5F0E8; 
                   text-shadow: 2px 2px 8px rgba(0,0,0,0.8);'>
            The Other Shelf
        </h1>
        <p style='font-size:2.5rem; color:#D4C5A9; 
                  text-shadow: 1px 1px 4px rgba(0,0,0,0.8);'>
            Discover books beyond the mainstream
        </p>
    </div>
""", unsafe_allow_html=True)

# ── Clickable books using columns ────────────────────────────────────────────
st.markdown("""
    <style>
        div[data-testid="column"] {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .book-img {
            height: 240px;
            width: auto;
            object-fit: contain;
            filter: drop-shadow(0 10px 25px rgba(0,0,0,0.7));
            transition: transform 0.35s ease;
            cursor: pointer;
        }
        .book-img:hover {
            transform: translateY(-18px) scale(1.06);
        }
        .book-title {
            color: #F5F0E8;
            font-size: 1.5rem;
            font-weight: bold;
            text-align: center;
            text-shadow: 1px 1px 6px rgba(0,0,0,0.9);
            margin-top: 0.75rem;
        }
        .book-subtitle {
            color: #D4C5A9;
            font-size: 1.2rem;
            text-align: center;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.8);
        }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <a href="/World_Fantasy" target="_self" style="text-decoration:none;">
            <div style="text-align:center;">
                <img class="book-img" src="data:image/png;base64,{img_fantasy}"/>
                <div class="book-title">World Fantasy</div>
                <div class="book-subtitle">Mythology & folklore from beyond the west</div>
            </div>
        </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <a href="/Non-Fiction" target="_self" style="text-decoration:none;">
            <div style="text-align:center;">
                <img class="book-img" src="data:image/png;base64,{img_nonfic}"/>
                <div class="book-title">Non-Fiction</div>
                <div class="book-subtitle">Left-wing thought & alternative perspectives</div>
            </div>
        </a>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <a href="/Graphic_Novels" target="_self" style="text-decoration:none;">
            <div style="text-align:center;">
                <img class="book-img" src="data:image/png;base64,{img_graphic}"/>
                <div class="book-title">Graphic Novels</div>
                <div class="book-subtitle">Sequential art beyond the mainstream</div>
            </div>
        </a>
    """, unsafe_allow_html=True)

# ── Shelf + footer ────────────────────────────────────────────────────────────
st.markdown("""
    <div style="height:6px; background:linear-gradient(180deg,#8B6914 0%,#5C4208 100%); 
                border-radius:2px; margin:6rem auto 0 auto;
                box-shadow:0 4px 8px rgba(0,0,0,0.5); max-width:680px;">
    </div>
    <p style='text-align:center; opacity:0.55; color:#D4C5A9; 
              margin-top:1rem; font-size:1rem; letter-spacing:0.03rem;'>
        Sarah Jane Nede &nbsp;·&nbsp; Gonçalo Trindade &nbsp;·&nbsp; Rachel Vianna &nbsp;— 2026
    </p>
""", unsafe_allow_html=True)