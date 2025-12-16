import sys
import os
import streamlit.web.cli as stcli

def resolve_path(path):
    """Resolve resource path for PyInstaller bundled apps."""
    if getattr(sys, "frozen", False):
        # If running as an exe, resources are in the temp folder _MEIPASS
        basedir = sys._MEIPASS
    else:
        # If running as a script, resources are in the current directory
        basedir = os.path.dirname(__file__)
    return os.path.join(basedir, path)

if __name__ == "__main__":
    # When frozen, we can't rely on 'streamlit run' command line.
    # We invoke the CLI programmatically.
    
    # 1. Point to the bundled app.py
    app_path = resolve_path("app.py")
    
    # 2. Mock command line arguments for Streamlit
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
    ]
    
    # 3. Launch Streamlit
    sys.exit(stcli.main())

