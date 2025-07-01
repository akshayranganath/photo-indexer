#!/usr/bin/env python3
"""
Standalone launcher for Photo-Indexer web UI.

Usage:
    python src/photo_indexer/ui/run_ui.py
    python -m photo_indexer.ui.run_ui
    
This script can be used to launch the Streamlit UI directly without
going through the CLI, useful for development or quick access.
"""

import sys
from pathlib import Path

def main():
    """Launch the Streamlit UI directly."""
    import subprocess
    
    # Get the path to the Streamlit app
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        print(f"❌ UI app not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Starting Photo-Indexer web UI...")
    print("📖 Open your browser to: http://localhost:8501")
    print("⚡ Press Ctrl+C to stop the server")
    print()
    
    try:
        # Launch Streamlit with reasonable defaults
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✨ Web UI stopped by user. Goodbye!")
    except Exception as exc:
        print(f"❌ Error running web UI: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main() 