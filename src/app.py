# src/app.py
import os
import io
import base64
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import cv2
from flask import Flask, request, render_template_string, redirect, url_for
import replicate
import requests
from flask import Flask, request, jsonify, render_template 
from flask import request, jsonify
import json

from huggingface_hub import InferenceClient

load_dotenv()

# ---------- AzureOpenAI client initialization (fix NameError) ----------
# make sure this import exists near the top of the file:
# from openai import AzureOpenAI
# ---------- Robust Azure client initialization ----------
# (paste this after load_dotenv() and imports)

# Try to import partner style AzureOpenAI (some openai versions expose it)
client = None
USE_OPENAI_FALLBACK = False

try:
    # prefer the partner-style client if available
    from openai import AzureOpenAI  # noqa
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_API_BASE = os.getenv("AZURE_API_BASE")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

    if AZURE_API_KEY and AZURE_API_BASE and AZURE_API_VERSION:
        client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_API_BASE,
        )
        print("✅ AzureOpenAI client (from openai.AzureOpenAI) initialized.")
    else:
        print("⚠️ AzureOpenAI env vars missing; will try openai fallback if configured.")
        client = None

except Exception as e_import:
    # fallback: use the openai package (common) — requires AZURE_DEPLOYMENT_NAME env
    try:
        import openai
        
        AZURE_API_KEY = os.getenv("AZURE_API_KEY")
        AZURE_API_BASE = os.getenv("AZURE_API_BASE")
        AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
        AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

        client = None

        if not (AZURE_API_KEY and AZURE_API_BASE and AZURE_API_VERSION and AZURE_DEPLOYMENT_NAME):
            print("⚠️ Missing Azure OpenAI env: please set AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION, AZURE_DEPLOYMENT_NAME in .env")
        else:
            # configure openai for Azure
            openai.api_type = "azure"
            openai.api_key = AZURE_API_KEY
            openai.api_base = AZURE_API_BASE.rstrip("/")
            openai.api_version = AZURE_API_VERSION
            print("ℹ️ openai configured for Azure. Using deployment:", AZURE_DEPLOYMENT_NAME)

            # tiny wrapper so your existing code can call client.chat.completions.create(...)
            class _WrappedCompletion:
                def __init__(self, resp_json):
                    self._resp = resp_json
                @property
                def choices(self):
                    class _Msg:
                        def __init__(self, text): self.content = text
                    class _Choice:
                        def __init__(self, text): self.message = _Msg(text)
                    try:
                        txt = self._resp["choices"][0]["message"]["content"]
                    except Exception:
                        txt = str(self._resp)
                    return [_Choice(txt)]

            class _ChatCompletions:
                def create(self, **kwargs):
                    # ensure engine arg for Azure
                    if "engine" not in kwargs:
                        kwargs["engine"] = AZURE_DEPLOYMENT_NAME
                    resp = openai.ChatCompletion.create(**kwargs)
                    return _WrappedCompletion(resp)

            class _ClientWrapper:
                def __init__(self):
                    self.chat = type("X", (), {})()
                    self.chat.completions = _ChatCompletions()

            client = _ClientWrapper()
        # --------------------------------------------------------------
            USE_OPENAI_FALLBACK = True

    except Exception as e_fallback:
        client = None
        print("⚠️ Failed to import AzureOpenAI and fallback openai setup failed.")
        print(f"Import error: {e_import}")
        print(f"Fallback error: {e_fallback}")
# -----------------------------------------------------------------


# ---------------------------
# Helpers
# ---------------------------

STABILITY_KEY = os.getenv("STABILITY_API_KEY")
if not STABILITY_KEY:
    raise RuntimeError("STABILITY_API_KEY missing in .env")

# allowed SDXL combos (from docs)
SDXL_ALLOWED = {
    (1024,1024), (1152,896), (896,1152),
    (1216,832), (1344,768), (768,1344),
    (1536,640), (640,1536)
}

def _ensure_size_for_engine(width:int, height:int, engine_id:str):
    """Validate or coerce width,height based on engine rules. Returns (w,h)."""
    if "xl" in engine_id.lower():
        if (width, height) in SDXL_ALLOWED:
            return width, height
        # default to 1024x1024 for SDXL if user choice isn't allowed
        return 1024, 1024
    # non-xl fallback (not used here) — ensure minimum pixels and multiples of 64
    min_pixels = 262_144
    if width * height < min_pixels:
        scale = (min_pixels / (width * height)) ** 0.5
        new_w = max(64, int(round(width * scale)))
        new_h = max(64, int(round(height * scale)))
        new_w = ((new_w + 63) // 64) * 64
        new_h = ((new_h + 63) // 64) * 64
        return new_w, new_h
    new_w = ((width + 63) // 64) * 64
    new_h = ((height + 63) // 64) * 64
    return new_w, new_h

def generate_image_from_prompt(prompt: str, size="1024x1024", model: str = None):
    """
    Stability SDXL text->image call. Returns PIL.Image or raises RuntimeError with details.
    NOTE: SDXL generation costs credits (30 steps or fewer -> 0.9 credits per generation).
    """
    engine = model or os.getenv("STABILITY_MODEL", "").strip()
    if not engine:
        raise RuntimeError("STABILITY_MODEL missing in .env (engine id)")

    # parse requested size
    try:
        width, height = map(int, size.split("x"))
    except Exception:
        width, height = 1024, 1024

    # validate/coerce size for this engine
    width, height = _ensure_size_for_engine(width, height, engine)

    url = f"https://api.stability.ai/v1/generation/{engine}/text-to-image"
    headers = {
        "Authorization": f"Bearer {STABILITY_KEY}",
        "Accept": "image/png",           # get raw PNG back
        "Content-Type": "application/json"
    }

    payload = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7.0,
        "height": height,
        "width": width,
        "samples": 1,
        "steps": 20                       # <=30 => 0.9 credits (smaller step saves time)
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
    except Exception as e:
        raise RuntimeError(f"Network error calling Stability API: {e}")

    if resp.status_code != 200:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise RuntimeError(f"Stability API error {resp.status_code}: {body}")

    # Parse PNG bytes
    try:
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return img
    except Exception as e:
        # fallback: check for base64 artifact
        try:
            j = resp.json()
            if "artifacts" in j and j["artifacts"]:
                b64 = j["artifacts"][0].get("base64")
                if b64:
                    img_bytes = base64.b64decode(b64)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    return img
            raise
        except Exception:
            raise RuntimeError(f"Failed to decode image: {e}; raw response length={len(resp.content)}")





def pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/png;base64," + b64

def extract_color_palette(pil_img: Image.Image, n_colors=4):
    img = pil_img.copy().resize((256, 256))
    arr = np.array(img).reshape(-1, 3).astype(float)
    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init=5)
    labels = kmeans.fit_predict(arr)
    centers = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    centers = centers[order]
    hexes = ["#%02x%02x%02x" % tuple(c) for c in centers]
    return hexes

def simple_style_classifier(prompt: str):
    p = prompt.lower()
    mapping = {
        "formal": ["suit", "blazer", "formal", "business", "office", "corporate"],
        "casual": ["casual", "tshirt", "jeans", "everyday", "relaxed"],
        "traditional": ["sherwani", "kurta", "sari", "saree", "traditional", "ethnic"],
        "party": ["party", "sequins", "glitter", "evening", "gown", "cocktail"],
        "streetwear": ["streetwear", "hoodie", "oversized", "street", "urban"],
        "sportswear": ["sportswear", "athletic", "gym", "running", "sport"],
    }
    scores = {}
    for label, keywords in mapping.items():
        scores[label] = sum(p.count(k) for k in keywords)
    label = max(scores, key=lambda k: scores[k])
    if scores[label] == 0:
        return "Casual / Other"
    return label.capitalize()

def accessory_recommender(prompt: str, style_label: str):
    p = prompt.lower()
    suggestions = []
    if style_label.lower().startswith("traditional"):
        suggestions = ["Embroidered mojari / ethnic shoes", "Statement turban / dupatta", "Gold-tone jewelry"]
    elif style_label.lower().startswith("formal"):
        suggestions = ["Oxford shoes", "Pocket square", "Minimal wristwatch", "Cufflinks"]
    elif style_label.lower().startswith("party"):
        suggestions = ["Strappy heels", "Clutch bag", "Chunky necklace", "Smoky makeup"]
    elif "beach" in p or "summer" in p:
        suggestions = ["Straw hat", "Sunglasses", "Sandals", "Shell jewelry"]
    elif style_label.lower().startswith("streetwear"):
        suggestions = ["Chunky sneakers", "Baseball cap", "Crossbody bag", "Layered chains"]
    else:
        suggestions = ["Neutral shoes", "Minimal jewelry", "Sunglasses"]
    return suggestions



app = Flask(__name__, static_folder="static", template_folder="templates")

# --- Initialize AzureOpenAI client if available (partner code style) ---
client = None
try:
    # try to import the AzureOpenAI wrapper your partner used
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
    )
    app.logger.info("✅ AzureOpenAI client (from openai.AzureOpenAI) initialized.")
except Exception as e:
    client = None
    app.logger.warning(f"⚠️ Failed to initialize AzureOpenAI client: {e}")

# --------------------------------------------------------------------------------
# Helper stubs — replace these with your real implementations (local pipeline or remote)
# --------------------------------------------------------------------------------


def pil_to_base64(img: Image.Image):
    """
    Convert a PIL image to a data URL (base64 png) for embedding into templates.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    import base64
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def extract_color_palette(img: Image.Image, n_colors: int = 4):
    """
    Return a list of hex color strings extracted from the image.
    Replace with an actual color quantization if desired.
    """
    # simple placeholder: return a set of neutral colors
    return ["#d6c9b7", "#9b8c7a", "#5a4f48", "#2b2b2b"][:n_colors]

def simple_style_classifier(prompt: str):
    """
    Very simple style classification from the prompt.
    Replace with a real classifier if you have one.
    """
    p = prompt.lower()
    if any(k in p for k in ["formal", "suit", "tux", "blazer", "gown"]):
        return "Formal"
    if any(k in p for k in ["casual", "jeans", "sneaker", "t-shirt", "hoodie"]):
        return "Casual"
    if any(k in p for k in ["ethnic", "sari", "kurta", "kimono", "hanbok"]):
        return "Ethnic"
    return "Contemporary"

def accessory_recommender(prompt: str, style_label: str):
    """
    Return a short list of accessory suggestions (strings).
    Replace with better logic / LLM if you want.
    """
    base = {
        "Formal": ["Pocket square", "Minimal wristwatch", "Cufflinks"],
        "Casual": ["Leather crossbody bag", "Gold hoops", "White sneakers"],
        "Ethnic": ["Jhumka earrings", "Embroidered clutch", "Kolhapuri sandals"],
        "Contemporary": ["Layered necklaces", "Sunglasses", "Ankle boots"],
    }
    return base.get(style_label, base["Contemporary"])

# --------------------------------------------------------------------------------
# /chatbot route: receives form field 'prompt' and returns JSON with 'raw' and 'parsed'
# --------------------------------------------------------------------------------
@app.route("/chatbot", methods=["POST"])
def flask_chatbot():
    """
    Accepts form field 'prompt'. Returns JSON:
    {
      "ok": true/false,
      "raw": "<raw model text>",
      "parsed": { enhanced_prompt, accessories, hairstyle, footwear, references }  # when available
    }
    """
    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"ok": False, "error": "Missing 'prompt' field."}), 400

    if client is None:
        return jsonify({"ok": False, "error": "AzureOpenAI client not configured."}), 500

    try:
        system_prompt = """
        You are a professional fashion stylist and AI prompt engineer.

        Your task:
        1. Rephrase the user's clothing request into a concise yet visually clear Stable Diffusion prompt (under 25 words).
        2. Suggest 2–3 matching accessories, 1–2 hairstyles, and 1–2 footwear options.
        3. Generate 1–2 reference shopping links (from real fashion e-commerce sites like Myntra, H&M, or Amazon) that best match the outfit. Use proper clickable Markdown format.

        Respond strictly in JSON format with keys:
        enhanced_prompt, accessories, hairstyle, footwear, references.
        """

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        # extract textual reply (partner style)
        try:
            reply_text = completion.choices[0].message.content
        except Exception:
            reply_text = str(completion)

        response_payload = {"ok": True, "raw": reply_text}

        # best-effort: try to parse JSON directly or strip fenced blocks
        parsed_obj = None
        try:
            cleaned = reply_text.strip()
            if cleaned.lower().startswith("```json"):
                # drop the first fence token
                cleaned = cleaned.split("```", 1)[1].strip()
            if cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = cleaned[3:-3].strip()
            cleaned = cleaned.strip('`"\' \n\r\t')
            parsed_obj = json.loads(cleaned)
            if isinstance(parsed_obj, dict):
                response_payload["parsed"] = parsed_obj
        except Exception:
            parsed_obj = None

        # secondary attempt: if top-level reply was raw JSON string
        if parsed_obj is None:
            try:
                maybe = json.loads(reply_text)
                if isinstance(maybe, dict):
                    response_payload["parsed"] = maybe
            except Exception:
                pass

        return jsonify(response_payload)
    except Exception as e:
        return jsonify({"ok": False, "error": f"AzureOpenAI error: {str(e)}"}), 500

# --------------------------------------------------------------------------------
# Index route using render_template('index.html', ...)
# Accepts POST from genForm with fields: prompt, size, variations, steps (optional)
# --------------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    # default variables for template
    prompt = ""
    image_b64 = None
    out_images = None
    palette = None
    rec_text = None
    style_label = None

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        size = request.form.get("size", "512x512")
        try:
            variations = int(request.form.get("variations", "1"))
        except Exception:
            variations = 1

        if not prompt:
            return render_template(
                "index.html",
                error="Please provide a prompt.",
                prompt=prompt,
                size=size,
                variations=str(variations),
            )

        try:
            # generate image (take first variation for now)
            img = generate_image_from_prompt(prompt, size=size)
            image_b64 = pil_to_base64(img)

            # auxiliary info for UI
            palette = extract_color_palette(img, n_colors=4)
            style_label = simple_style_classifier(prompt)
            recs = accessory_recommender(prompt, style_label)
            rec_text = "\n".join(f"- {r}" for r in recs)

            # optionally prepare out_images list if you generate multiple variations
            out_images = [image_b64]  # currently one; replace with list of base64 or URLs

            return render_template(
                "index.html",
                prompt=prompt,
                image_b64=image_b64,
                out_images=out_images,
                palette=palette,
                rec_text=rec_text,
                style_label=style_label,
                size=size,
                variations=str(variations),
            )
        except Exception as e:
            return render_template("index.html", error=str(e), prompt=prompt, size=size)

    # GET
    return render_template("index.html", prompt=prompt, size="512x512", variations="1")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 7860)), debug=False)

