# src/check_model.py
import os, requests
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_API_TOKEN")
MODEL = os.getenv("HF_MODEL", "runwayml/stable-diffusion-v1-5")

if not HF_TOKEN:
    raise SystemExit("HF_API_TOKEN missing in .env")

url = f"https://huggingface.co/api/models/{MODEL}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
resp = requests.get(url, headers=headers, timeout=30)

print("REQUEST:", url)
print("STATUS:", resp.status_code)
try:
    print("BODY:", resp.json())
except Exception:
    print("BODY (text):", resp.text)
