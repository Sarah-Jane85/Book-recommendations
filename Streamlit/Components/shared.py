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

def back_button():
    """Renders a back to home button."""
    st.markdown("""
        <a href="/" target="_self" style="
            display: inline-block;
            color: #D4C5A9;
            text-decoration: none;
            font-size: 0.9rem;
            padding: 0.4rem 0.8rem;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 6px;
            background: rgba(0,0,0,0.3);
            margin-bottom: 1rem;
            transition: all 0.2s ease;">
            ← Home
        </a>
    """, unsafe_allow_html=True)

# ── Author bio ────────────────────────────────────────────────────────────────
import requests

@st.cache_data
@st.cache_data
def get_author_bio(author_name):
    import requests
    headers = {"User-Agent": "TheOtherShelf/1.0 (book recommender project)"}
    try:
        r = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" +
            author_name.replace(" ", "_"),
            headers=headers,
            timeout=5
        )
        if r.status_code == 200:
            data = r.json()
            extract = data.get("extract", "")
            # Make sure it's actually about a person/author
            # not a name definition, place, or disambiguation
            author_keywords = ["author", "writer", "novelist", "poet", 
                              "fiction", "fantasy", "science fiction",
                              "born", "published", "book", "novel"]
            extract_lower = extract.lower()
            is_relevant = any(kw in extract_lower for kw in author_keywords)
            
            if is_relevant and data.get("type") == "standard":
                return {
                    "extract": extract,
                    "image":   data.get("thumbnail", {}).get("source", ""),
                    "url":     data.get("content_urls", {}).get("desktop", {}).get("page", "")
                }
        return None
    except:
        return None

def show_author_bio(authors, safe_fn):
    selected = st.selectbox(
        "author_select",
        ["— select an author —"] + authors,
        label_visibility="collapsed",
        key="author_bio_select"
    )

    if selected != "— select an author —":
        with st.spinner("Loading bio..."):
            bio = get_author_bio(selected)
        if bio and bio["extract"]:
            col_text, col_img = st.columns([3, 1])  # text left, image right
            with col_text:
                st.markdown(f"""
                    <div style="background:rgba(0,0,0,0.7); padding:1rem 1.25rem;
                                border-radius:10px; max-width:750px; margin:0 auto;">
                        <h4 style="color:#F5D78E; margin:0 0 0.5rem 0;">
                            {safe_fn(selected)}
                        </h4>
                        <p style="color:#F5F0E8; font-size:0.9rem; line-height:1.7;">
                            {safe_fn(bio['extract'][:500])}...
                        </p>
                        <a href="{bio['url']}" target="_blank"
                           style="color:#A8D5B5; font-size:0.85rem;">
                            Read more on Wikipedia →
                        </a>
                    </div>
                """, unsafe_allow_html=True)
            with col_img:
                if bio["image"]:
                    st.image(bio["image"], width=130)
        else:
            st.markdown(f"""
                <p style="color:#D4C5A9; font-size:0.9rem;">
                    No Wikipedia page found for {safe_fn(selected)}.
                </p>
            """, unsafe_allow_html=True)