# src/list_stability_engines.py
import os, requests
from dotenv import load_dotenv
load_dotenv()

KEY = os.getenv("STABILITY_API_KEY")
if not KEY:
    raise SystemExit("STABILITY_API_KEY missing in .env")

url = "https://api.stability.ai/v1/engines/list"
headers = {"Authorization": f"Bearer {KEY}"}
resp = requests.get(url, headers=headers, timeout=30)
print("STATUS:", resp.status_code)
try:
    print(resp.json())
except Exception:
    print(resp.text)
