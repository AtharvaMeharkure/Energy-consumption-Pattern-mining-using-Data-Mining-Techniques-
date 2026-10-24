"""
VoltPulse AI - One-Click Application Launcher
Starts the full-stack FastAPI server and opens the browser.
"""
import sys
import os
import time
import webbrowser
import threading
import uvicorn

def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    print("=" * 65)
    print("⚡ VOLTPULSE AI - Energy Consumption Pattern Mining Platform")
    print("=" * 65)
    print("Starting backend server at http://127.0.0.1:8000 ...")
    print("Press Ctrl+C to terminate the server.\n")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
