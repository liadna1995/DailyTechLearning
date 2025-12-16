import os
import sys
import platform
import subprocess
from dotenv import load_dotenv

def setup_scheduler():
    load_dotenv()
    
    popup_time = os.getenv("POPUP_TIME", "09:00")
    
    # Validate time format
    try:
        hour, minute = popup_time.split(":")
        hour = int(hour)
        minute = int(minute)
    except ValueError:
        print("❌ Invalid POPUP_TIME format in .env. Use HH:MM (24-hour format).")
        return

    cwd = os.getcwd()
    system = platform.system()
    task_name = "DailyTechLearning"
    
    # DETECT EXECUTABLE
    if getattr(sys, 'frozen', False):
        # If this script is compiled as an exe, sys.executable is SetupScheduler.exe
        # We want to target the main app in the same folder
        base_dir = os.path.dirname(sys.executable)
        exe_name = "DailyTechLearning.exe" if system == "Windows" else "DailyTechLearning"
        target_app = os.path.join(base_dir, exe_name)
    else:
        # If running from python, look for the built exe in dist, or default to current python setup
        # But per user request, we assume we are scheduling the EXE.
        
        exe_name = "DailyTechLearning.exe" if system == "Windows" else "DailyTechLearning"
        
        # Check if the exe exists in current folder (deployed) or dist (dev)
        if os.path.exists(os.path.join(cwd, exe_name)):
            target_app = os.path.join(cwd, exe_name)
        elif os.path.exists(os.path.join(cwd, "dist", exe_name)):
            target_app = os.path.join(cwd, "dist", exe_name)
        else:
            print(f"⚠️  Could not find '{exe_name}'. Using python script fallback.")
            # Fallback to the bat/sh scripts if exe not found
            target_app = os.path.join(cwd, "run_windows.bat" if system == "Windows" else "run_mac.sh")
            
    print(f"🔧 Configuring daily popup for {popup_time} on {system}...")
    print(f"📂 Target Application: {target_app}")

    if system == "Windows":
        # Using PowerShell to create a task with "StartWhenAvailable" (Run ASAP if missed)
        working_dir = os.path.dirname(target_app)
        
        ps_command = f"""
        $Action = New-ScheduledTaskAction -Execute "{target_app}" -WorkingDirectory "{working_dir}"
        $Trigger = New-ScheduledTaskTrigger -Daily -At {popup_time}
        $Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
        Register-ScheduledTask -TaskName "{task_name}" -Action $Action -Trigger $Trigger -Settings $Settings -Force
        """
        
        try:
            # Save PS command to temp file to avoid quoting issues
            with open("create_task.ps1", "w") as f:
                f.write(ps_command)
            
            subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", "create_task.ps1"], check=True)
            os.remove("create_task.ps1")
            
            print(f"✅ Success! Task '{task_name}' created/updated.")
            print(f"   - Scheduled for {popup_time} daily.")
            print(f"   - Configured to run ASAP if the computer was off at that time.")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating task via PowerShell: {e}")
            print("👉 Try running this script as Administrator.")

    elif system == "Darwin": # macOS
        # Create a LaunchAgent plist for launchd
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.dailytech.learning</string>
    <key>ProgramArguments</key>
    <array>
        <string>{target_app}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{os.path.dirname(target_app)}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{os.path.join(cwd, "launchd.log")}</string>
    <key>StandardErrorPath</key>
    <string>{os.path.join(cwd, "launchd_error.log")}</string>
</dict>
</plist>
"""
        home_dir = os.path.expanduser("~")
        launch_agents_dir = os.path.join(home_dir, "Library", "LaunchAgents")
        os.makedirs(launch_agents_dir, exist_ok=True)
        
        plist_path = os.path.join(launch_agents_dir, "com.dailytech.learning.plist")
        
        with open(plist_path, "w") as f:
            f.write(plist_content)
            
        try:
            subprocess.run(["launchctl", "unload", plist_path], stderr=subprocess.DEVNULL)
            subprocess.run(["launchctl", "load", plist_path], check=True)
            
            print(f"✅ Success! LaunchAgent created at {plist_path}")
            print(f"   - Scheduled for {popup_time} daily.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error loading launchd task: {e}")

    else:
        print(f"❌ Unsupported operating system: {system}")

if __name__ == "__main__":
    setup_scheduler()
