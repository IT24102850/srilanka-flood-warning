import os
import sys
import time
import threading
import subprocess
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

def start_streamlit():
    print("🚀 Starting Streamlit Dashboard on port 8501...")
    # Run Streamlit in headless mode so it doesn't open an extra browser tab automatically
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/app.py", "--server.port", "8501", "--server.headless", "true"])

def start_landing_page():
    print("🌐 Starting Landing Page Server on port 8000...")
    # Serve from the root directory so the outside 'images' folder is accessible
    server = HTTPServer(("localhost", 8000), SimpleHTTPRequestHandler)
    server.serve_forever()

if __name__ == "__main__":
    # Start both servers in background threads
    threading.Thread(target=start_streamlit, daemon=True).start()
    threading.Thread(target=start_landing_page, daemon=True).start()
    
    # Give servers a few seconds to boot up
    time.sleep(3)
    
    print("✅ Opening Landing Page in your browser...")
    webbrowser.open("http://localhost:8000/app/landing.html")
    
    print("\nPress Ctrl+C to stop both servers.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")