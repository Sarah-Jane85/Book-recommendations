import sys
from pathlib import Path
import html
import requests
import time
import base64

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent.parent))
from Src.recommender import build_index, load_data, recommend
from Components.shared import set_page_style, show_author_bio, back_button


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cover_url(title, author):
    """Get book cover URL from Open Library API."""
    try:
        # Search Open Library for the book
        search_url = "https://openlibrary.org/search.json"
        # Use general query instead of separate title/author for better results
        query = f"{title} {author}"
        params = {
            "q": query,
            "limit": 1
        }
        response = requests.get(search_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get("docs"):
            doc = data["docs"][0]
            # Try to get cover from various sources
            if "cover_i" in doc:
                cover_url = f"https://covers.openlibrary.org/b/id/{doc['cover_i']}-M.jpg"
                # Quick check if the cover URL is accessible
                try:
                    cover_response = requests.head(cover_url, timeout=2)
                    if cover_response.status_code == 200:
                        return cover_url
                except:
                    pass  # Fall through to placeholder
            elif "isbn" in doc and doc["isbn"]:
                cover_url = f"https://covers.openlibrary.org/b/isbn/{doc['isbn'][0]}-M.jpg"
                try:
                    cover_response = requests.head(cover_url, timeout=2)
                    if cover_response.status_code == 200:
                        return cover_url
                except:
                    pass  # Fall through to placeholder
        
        # Fallback: return a data URL for a simple book icon
        # This is a base64 encoded SVG that will always work
        book_svg = '''<svg width="100" height="150" xmlns="http://www.w3.org/2000/svg">
            <rect width="100" height="150" fill="#e8d5b7" stroke="#8b4513" stroke-width="2"/>
            <rect x="10" y="20" width="80" height="110" fill="#f5f0e8"/>
            <text x="50" y="80" text-anchor="middle" font-family="serif" font-size="12" fill="#8b4513">Book</text>
            <line x1="10" y1="35" x2="90" y2="35" stroke="#8b4513" stroke-width="1"/>
            <line x1="10" y1="50" x2="90" y2="50" stroke="#8b4513" stroke-width="1"/>
            <line x1="10" y1="65" x2="90" y2="65" stroke="#8b4513" stroke-width="1"/>
        </svg>'''
        svg_base64 = base64.b64encode(book_svg.encode('utf-8')).decode('utf-8')
        return f"data:image/svg+xml;base64,{svg_base64}"
        
    except Exception as e:
        # Return the same SVG placeholder on error
        book_svg = '''<svg width="100" height="150" xmlns="http://www.w3.org/2000/svg">
            <rect width="100" height="150" fill="#e8d5b7" stroke="#8b4513" stroke-width="2"/>
            <rect x="10" y="20" width="80" height="110" fill="#f5f0e8"/>
            <text x="50" y="80" text-anchor="middle" font-family="serif" font-size="12" fill="#8b4513">Book</text>
            <line x1="10" y1="35" x2="90" y2="35" stroke="#8b4513" stroke-width="1"/>
            <line x1="10" y1="50" x2="90" y2="50" stroke="#8b4513" stroke-width="1"/>
            <line x1="10" y1="65" x2="90" y2="65" stroke="#8b4513" stroke-width="1"/>
        </svg>'''
        import base64
        svg_base64 = base64.b64encode(book_svg.encode('utf-8')).decode('utf-8')
        return f"data:image/svg+xml;base64,{svg_base64}"


st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")
set_page_style()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 1rem 0;">
        <h1 style="font-size:3rem; color:#F5F0E8;
                   text-shadow:2px 2px 8px rgba(0,0,0,0.8);">
            📚 Book Recommender
        </h1>
        <p style="font-size:1.05rem; color:#D4C5A9; margin-bottom:0.75rem;">
            Non-Fiction & Alternative Perspectives
        </p>
        <p style="font-size:1rem; color:#F5F0E8; max-width:700px;
                  margin:0 auto; line-height:1.8;
                  background:rgba(0,0,0,0.5); padding:1rem 1.5rem;
                  border-radius:10px;">
            Search by keyword, topic, author, or title — we'll find the closest matches
            in our collection of left-wing thought and alternative perspectives.
            <strong style="color:#F5D78E;">3,034 books</strong>
            from critical theory, postcolonial studies, and radical perspectives.
        </p>
    </div>
    <hr style="border-color:rgba(255,255,255,0.15); margin-bottom:1.5rem;">
""", unsafe_allow_html=True)

back_button()

# Improve text input visibility
st.markdown("""
    <style>
        .stTextInput > div > div > input {
            background-color: rgba(100,100,100,0.5) !important;
            color: #F5F0E8 !important;
        }
        .stTextInput > div > div > input::placeholder {
            color: rgba(245,240,232,0.7) !important;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_index(_df):
    return build_index(_df)


df = get_data()
vectorizer, matrix = get_index(df)

query = st.text_input(
    "What are you looking for?",
    placeholder="e.g. capitalism, colonialism, David Harvey, Capital...",
)
top_n = st.slider("Number of results", min_value=3, max_value=20, value=10)

if query.strip():
    results = recommend(query, df, vectorizer, matrix, top_n=top_n)

    if results.empty:
        st.info("No matches found. Try different keywords.")
    else:
        st.markdown(f"**{len(results)} results** for *{query}*")
        st.divider()

        for _, row in results.iterrows():
            year = f" · {int(row['year_published'])}" if pd.notna(row["year_published"]) else ""
            
            # Get cover URL
            cover_url = get_cover_url(row['title'], row['author'])
            
            # Create columns for cover and text
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.image(cover_url, width=100, caption="")
            
            with col2:
                st.markdown(f"### {row['title']}")
                st.markdown(f"**{row['author']}**{year}")
                
                # Add Goodreads link if available
                if pd.notna(row.get('open_library_key')) and row['open_library_key']:
                    # Extract Goodreads ID from Open Library key (format: /book/show/ID.title)
                    ol_key = str(row['open_library_key'])
                    if '/book/show/' in ol_key:
                        try:
                            goodreads_id = ol_key.split('/book/show/')[1].split('.')[0]
                            goodreads_url = f"https://www.goodreads.com/book/show/{goodreads_id}"
                            st.markdown(f"[📖 View on Goodreads]({goodreads_url})")
                        except:
                            pass
                
                with st.expander("Description"):
                    st.write(row["description"])
            
            st.divider()
        
        st.markdown("### 📖 Learn More About the Authors")
        authors = results['author'].unique().tolist()
        show_author_bio(authors, html.escape)
