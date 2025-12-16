import PyInstaller.__main__
import os
import shutil
import platform

# Determine the OS separator for path definitions
sep = ';' if platform.system() == "Windows" else ':'

print("🚀 Starting Build Process...")

# Clean previous builds
if os.path.exists("dist"):
    shutil.rmtree("dist")
if os.path.exists("build"):
    shutil.rmtree("build")

print("📦 Building Main App (DailyTechLearning)...")
PyInstaller.__main__.run([
    'main.py',                        # Entry point
    '--name=DailyTechLearning',       # Executable name
    '--onefile',                      # Single executable file
    '--clean',                        # Clean cache
    f'--add-data=app.py{sep}.',       # Include app.py inside the exe
    
    # Collect all dependencies for packages that use lazy loading or metadata
    '--collect-all=streamlit',
    '--collect-all=google.genai',
    '--collect-all=edge_tts',
    '--collect-all=altair',           # Often needed by Streamlit
    '--collect-all=pandas',           # Often needed by Streamlit
])

print("\n📦 Building Scheduler Setup (SetupScheduler)...")
PyInstaller.__main__.run([
    'setup_scheduler.py',
    '--name=SetupScheduler',
    '--onefile',
    '--clean',
])

print("\n✅ Build Complete!")
print("📂 Your executables are in the 'dist' folder:")
print("   1. DailyTechLearning (The App)")
print("   2. SetupScheduler (The Installer/Configurator)")
print("\n👉 To distribute: Send the 'dist' folder content + .env + context.toml")
