"""
photo_indexer.ui.app
~~~~~~~~~~~~~~~~~~~~

Streamlit web application for searching and browsing indexed photos.

Usage:
    streamlit run src/photo_indexer/ui/app.py
"""

import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from photo_indexer.config import load_config, IndexerSettings
from photo_indexer.workers import _read_nef_thumbnail


# --------------------------------------------------------------------------- #
# Configuration & Constants                                                   #
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Photo Indexer",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
if "config" not in st.session_state:
    st.session_state.config = load_config()

config: IndexerSettings = st.session_state.config

# Thumbnail cache directory
THUMBNAIL_CACHE_DIR = Path("data/thumbnails").expanduser()
THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Database Operations                                                         #
# --------------------------------------------------------------------------- #

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_database_stats() -> Dict[str, Any]:
    """Get basic statistics about the photo database."""
    db_path = config.db_path
    print(f"Database path: {db_path}")
    if not db_path.exists():
        return {"total_photos": 0, "error": "Database not found"}
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Total photos
            cursor.execute("SELECT COUNT(*) FROM photos")
            total = cursor.fetchone()[0]
            
            # Photos with people
            cursor.execute("SELECT COUNT(*) FROM photos WHERE people = 1")
            with_people = cursor.fetchone()[0]
            
            # Date range
            cursor.execute("SELECT MIN(date), MAX(date) FROM photos")
            date_range = cursor.fetchone()
            
            # Scene distribution
            cursor.execute("SELECT scene, COUNT(*) FROM photos GROUP BY scene")
            scenes = dict(cursor.fetchall())
            
            return {
                "total_photos": total,
                "with_people": with_people,
                "date_range": date_range,
                "scenes": scenes,
            }
    except Exception as e:
        return {"total_photos": 0, "error": str(e)}


@st.cache_data(ttl=60)  # Cache for 1 minute
def search_photos(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Search photos using FTS5 full-text search."""
    db_path = config.db_path
    if not db_path.exists():
        return []
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Return dict-like rows
            cursor = conn.cursor()
            
            if query.strip():
                # Use FTS5 search if available, otherwise simple LIKE search
                try:
                    # First try FTS5 on the virtual table
                    cursor.execute("""
                        SELECT p.* FROM photos p
                        JOIN photos_fts fts ON p.file = fts.file
                        WHERE photos_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                    """, (query, limit))
                except sqlite3.OperationalError:
                    # Fallback to LIKE search on main table
                    cursor.execute("""
                        SELECT * FROM photos 
                        WHERE description LIKE ? 
                           OR location LIKE ?
                           OR scene LIKE ?
                        ORDER BY date DESC, time DESC
                        LIMIT ?
                    """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
            else:
                # Return all photos when no query
                cursor.execute("""
                    SELECT * FROM photos 
                    ORDER BY date DESC, time DESC 
                    LIMIT ?
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
            
    except Exception as e:
        st.error(f"Database error: {e}")
        return []


# --------------------------------------------------------------------------- #
# Thumbnail Management                                                        #
# --------------------------------------------------------------------------- #

def get_thumbnail_path(nef_path: str) -> Path:
    """Generate thumbnail cache path for a NEF file."""
    # Create a hash of the full path for unique filename
    path_hash = hashlib.md5(nef_path.encode()).hexdigest()
    return THUMBNAIL_CACHE_DIR / f"{path_hash}.jpg"


@st.cache_data(persist=True)  # Persist across sessions
def get_or_create_thumbnail(nef_path: str, thumb_size: int = 256) -> Optional[str]:
    """Get cached thumbnail or create new one from NEF file."""
    nef_file = Path(nef_path)
    
    # Check if NEF file exists
    if not nef_file.exists():
        return None
    
    # Check cache first
    thumb_path = get_thumbnail_path(nef_path)
    
    # Check if cached thumbnail exists and is newer than NEF
    if (thumb_path.exists() and 
        thumb_path.stat().st_mtime >= nef_file.stat().st_mtime):
        return str(thumb_path)
    
    try:
        # Generate new thumbnail
        img, _ = _read_nef_thumbnail(nef_file, thumb_px=thumb_size)
        
        # Save to cache
        img.save(thumb_path, "JPEG", quality=85, optimize=True)
        
        return str(thumb_path)
        
    except Exception as e:
        st.error(f"Error creating thumbnail for {nef_file.name}: {e}")
        return None


# --------------------------------------------------------------------------- #
# UI Components                                                               #
# --------------------------------------------------------------------------- #

def render_sidebar():
    """Render the sidebar with search controls and stats."""
    st.sidebar.title("📷 Photo Indexer")
    
    # Database stats
    with st.sidebar.expander("📊 Database Stats", expanded=True):
        stats = get_database_stats()
        
        if "error" in stats:
            st.error(f"❌ {stats['error']}")
            st.info("💡 Run indexing first: `pi index /path/to/photos`")
        else:
            st.metric("Total Photos", stats["total_photos"])
            st.metric("With People", stats["with_people"])
            
            if stats["date_range"][0]:
                st.write(f"📅 **Date Range:** {stats['date_range'][0]} to {stats['date_range'][1]}")
            
            if stats["scenes"]:
                st.write("🏞️ **Scenes:**")
                for scene, count in stats["scenes"].items():
                    st.write(f"  • {scene.title()}: {count}")
    
    # Search settings
    with st.sidebar.expander("⚙️ Settings", expanded=False):
        st.write(f"**Database:** `{config.db_path}`")
        st.write(f"**Thumbnail Cache:** `{THUMBNAIL_CACHE_DIR}`")
        
        if st.button("🗑️ Clear Thumbnail Cache"):
            clear_thumbnail_cache()
        
        if st.button("🔄 Refresh Database Stats"):
            get_database_stats.clear()
            st.rerun()


def clear_thumbnail_cache():
    """Clear all cached thumbnails."""
    try:
        for thumb_file in THUMBNAIL_CACHE_DIR.glob("*.jpg"):
            thumb_file.unlink()
        st.success("✅ Thumbnail cache cleared!")
        # Clear Streamlit cache
        get_or_create_thumbnail.clear()
    except Exception as e:
        st.error(f"❌ Error clearing cache: {e}")


def render_photo_grid(photos: List[Dict[str, Any]], columns: int = 4):
    """Render photos in a responsive grid layout."""
    if not photos:
        st.info("🔍 No photos found. Try a different search term or check if photos are indexed.")
        return
    
    st.write(f"**Found {len(photos)} photos**")
    
    # Create columns for grid layout
    for i in range(0, len(photos), columns):
        cols = st.columns(columns)
        
        for j, col in enumerate(cols):
            if i + j < len(photos):
                photo = photos[i + j]
                render_photo_card(col, photo)


def render_photo_card(col, photo: Dict[str, Any]):
    """Render a single photo card in the grid."""
    with col:
        # Get thumbnail
        thumb_path = get_or_create_thumbnail(photo["file"], thumb_size=256)
        
        if thumb_path and Path(thumb_path).exists():
            # Display thumbnail
            image = Image.open(thumb_path)
            st.image(image, use_container_width=True)
        else:
            # Placeholder for missing thumbnail
            st.error("❌ Thumbnail unavailable")
        
        # Photo metadata
        with st.expander(f"📝 {Path(photo['file']).name}", expanded=False):
            st.write(f"**Description:** {photo['description']}")
            st.write(f"**Scene:** {photo['scene'].title()}")
            st.write(f"**Location:** {photo['location'].title()}")
            st.write(f"**Date:** {photo['date']} at {photo['time']}")
            
            if photo['people']:
                people_emoji = "👥" if photo['count'] > 1 else "👤"
                st.write(f"**People:** {people_emoji} {photo['count']} person(s)")
            else:
                st.write("**People:** None detected")
            
            # Show file path
            st.code(photo["file"], language=None)


# --------------------------------------------------------------------------- #
# Main Application                                                            #
# --------------------------------------------------------------------------- #

def main():
    """Main Streamlit application."""
    # Render sidebar
    render_sidebar()
    
    # Main content area
    st.title("🔍 Photo Search")
    st.markdown("Search your indexed photos by description, location, or scene.")
    
    # Search input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search photos",
            placeholder="e.g., 'mountain sunset', 'beach family', 'indoor birthday'",
            help="Search by description, location, or scene. Leave empty to show all photos."
        )
    
    with col2:
        max_results = st.selectbox(
            "Max results",
            options=[20, 50, 100, 200],
            index=1,
            help="Maximum number of photos to display"
        )
    
    # Search and display results
    if st.button("🔍 Search") or query:
        with st.spinner("Searching photos..."):
            photos = search_photos(query, limit=max_results)
            render_photo_grid(photos)
    else:
        # Show recent photos by default
        st.subheader("📷 Recent Photos")
        with st.spinner("Loading recent photos..."):
            recent_photos = search_photos("", limit=20)
            render_photo_grid(recent_photos, columns=5)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip:** Use keywords like 'outdoor', 'indoor', 'people', 'mountain', 'beach', etc. "
        "for better search results."
    )


if __name__ == "__main__":
    main() 