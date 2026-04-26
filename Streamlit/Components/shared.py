# ── shared.py — styling and layout shared across all pages ──────────────────
import streamlit as st

BACKGROUND_URL = "https://www.pixelstalk.net/wp-content/uploads/2016/08/HD-Library-Wallpaper.jpg"

def set_page_style():
    """Call this at the top of every page to apply shared styling."""
    st.markdown(f"""
        <style>
            /* ── Background image ── */
            .stApp {{
                background-image: url("{BACKGROUND_URL}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}

            /* ── Dark overlay so text is readable ── */
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0; left: 0;
                width: 100%; height: 100%;
                background: rgba(0, 0, 0, 0.55);
                z-index: 0;
            }}

            /* ── Make all content sit above the overlay ── */
            .block-container {{
                position: relative;
                z-index: 1;
            }}

            /* ── Text colors ── */
            h1, h2, h3, h4, p, label, .stMarkdown {{
                color: #F5F0E8 !important;
            }}

            /* ── Input fields ── */
            .stTextInput > div > div > input {{
                background-color: rgba(255,255,255,0.15);
                color: #F5F0E8;
                border: 1px solid rgba(255,255,255,0.3);
                border-radius: 8px;
            }}

            /* ── Buttons ── */
            .stButton > button {{
                background-color: rgba(180, 130, 70, 0.8);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.5rem 1.5rem;
                font-weight: bold;
            }}
            .stButton > button:hover {{
                background-color: rgba(180, 130, 70, 1.0);
            }}

            /* ── Sidebar ── */
            .css-1d391kg, [data-testid="stSidebar"] {{
                background-color: rgba(20, 15, 10, 0.85) !important;
            }}
        </style>
    """, unsafe_allow_html=True)


def page_header(title, subtitle=None):
    """Renders a consistent header for each page."""
    st.markdown(f"# 📚 {title}")
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.markdown("---")