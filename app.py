import os
import io
import time
import tempfile
import hashlib
import base64
import threading
import wave

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from cachetools import TTLCache
import requests

# We import whisper and piper inside the lazy loaders to prevent blocking on startup.
# import whisper
# from piper import PiperVoice

load_dotenv()

app = Flask(__name__, template_folder="templates")
CORS(app)

# =========================
# Config
# =========================
VISION_MODEL = "qwen3-vl:2b-instruct-q4_K_M"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

EN_PIPER_MODEL_PATH = "models/en_US-amy-medium.onnx"
FR_PIPER_MODEL_PATH = "models/fr_FR-siwis-medium.onnx"

# Thread-safe Cache Settings (FIX: Replaced custom OrderedDict)
DETECT_CACHE_TTL = 12  # seconds
DETECT_CACHE_MAX_ITEMS = 100
detect_cache = TTLCache(maxsize=DETECT_CACHE_MAX_ITEMS, ttl=DETECT_CACHE_TTL)
cache_lock = threading.Lock()

# Detect cooldown settings
DETECT_REQUEST_COOLDOWN_SEC = 2.0
last_detect_time_by_ip = TTLCache(maxsize=1000, ttl=60) # Auto-cleans old IPs
ip_lock = threading.Lock()


# =========================
# Lazy Loading Singleton Managers (FIX: Prevents blocking on boot)
# =========================
_whisper_model = None
whisper_lock = threading.Lock()

def get_whisper():
    """Lazy loads the Whisper model only on the first STT request."""
    global _whisper_model
    with whisper_lock:
        if _whisper_model is None:
            print("Loading Whisper (base) for the first time...")
            start = time.time()
            import whisper
            _whisper_model = whisper.load_model("base")
            print(f"Whisper loaded in {time.time() - start:.2f} sec.")
    return _whisper_model

_voices = {}
voice_lock = threading.Lock()

def get_piper_voice(lang):
    """Lazy loads the requested Piper voice only on the first TTS request."""
    with voice_lock:
        if lang not in _voices:
            from piper import PiperVoice
            print(f"Loading Piper voice for '{lang}' for the first time...")
            start = time.time()
            if lang == "en" and os.path.exists(EN_PIPER_MODEL_PATH):
                _voices["en"] = PiperVoice.load(EN_PIPER_MODEL_PATH)
            elif lang == "fr" and os.path.exists(FR_PIPER_MODEL_PATH):
                _voices["fr"] = PiperVoice.load(FR_PIPER_MODEL_PATH)
            else:
                raise RuntimeError(f"Model path for '{lang}' not found.")
            print(f"Piper voice '{lang}' loaded in {time.time() - start:.2f} sec.")
    return _voices.get(lang)


# =========================
# Helper: pretty perf log
# =========================
def perf(label: str, start_time: float):
    elapsed = time.time() - start_time
    print(f"[PERF] {label}: {elapsed:.2f} sec")
    return elapsed


# =========================
# Detect helpers
# =========================
def normalize_question(q: str) -> str:
    return " ".join(q.lower().strip().split())


def image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def make_detect_cache_key(question: str, lang: str, image_bytes: bytes) -> str:
    q = normalize_question(question)
    img_h = image_hash(image_bytes)
    return f"{lang}:{q}:{img_h}"


def is_low_value_question(q: str) -> bool:
    qn = normalize_question(q)
    low_value_set = {"what", "huh", "ok", "hello", "test", "hey", "hmm"}
    return len(qn) < 3 or qn in low_value_set


def get_client_ip():
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def detect_cooldown_active():
    ip = get_client_ip()
    now = time.time()
    
    with ip_lock:
        last_time = last_detect_time_by_ip.get(ip)
        if last_time is not None and (now - last_time) < DETECT_REQUEST_COOLDOWN_SEC:
            return True, round(DETECT_REQUEST_COOLDOWN_SEC - (now - last_time), 2)
        last_detect_time_by_ip[ip] = now
    return False, 0


def call_ollama_vision(image_bytes: bytes, prompt: str) -> str:
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 60
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)

    if response.status_code == 404:
        raise RuntimeError(
            f"Model '{VISION_MODEL}' not found in Ollama. Run: ollama pull {VISION_MODEL}"
        )

    response.raise_for_status()
    data = response.json()
    return (data.get("response") or "").strip()


# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "vision_backend": "ollama",
        "vision_model": VISION_MODEL,
        "tts_model": "piper-local",
        "supported_languages": ["en", "fr"],
        "detect_cache_ttl_sec": DETECT_CACHE_TTL,
    })


@app.route("/stt", methods=["POST"])
def stt():
    """Speech-to-Text using local Whisper"""
    request_start = time.time()
    tmp_path = None

    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file sent"}), 400

        file = request.files["audio"]

        # Note: Whisper requires a file path because ffmpeg is used under the hood.
        # We must write to temp disk here unless migrating to faster-whisper.
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        whisper_start = time.time()
        model = get_whisper() # Lazy load integration
        result = model.transcribe(tmp_path)
        whisper_time = perf("STT - Whisper inference", whisper_start)

        text = result["text"].strip()
        total_time = perf("STT - total request", request_start)
        
        print(f"[STT] {text}")

        return jsonify({
            "text": text,
            "perf": {
                "whisper_sec": round(whisper_time, 2),
                "total_sec": round(total_time, 2),
            }
        })

    except Exception as e:
        print(f"[STT ERROR] {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/detect", methods=["POST"])
def detect():
    """Question answering from image."""
    request_start = time.time()

    try:
        cooldown_active, retry_after = detect_cooldown_active()
        if cooldown_active:
            return jsonify({
                "error": f"Please wait {retry_after} sec before asking again.",
                "retryable": True
            }), 429

        if "image" not in request.files:
            return jsonify({"error": "No image file sent"}), 400

        question = request.form.get("question", "").strip()
        lang = request.form.get("lang", "en").strip().lower()

        if not question or is_low_value_question(question):
            return jsonify({"error": "Question too short or unclear"}), 400

        if lang not in ("en", "fr"):
            lang = "en"

        image_bytes = request.files["image"].read()
        if not image_bytes:
            return jsonify({"error": "Empty image received"}), 400

        # FIX: Thread-safe Cache Check
        cache_key = make_detect_cache_key(question, lang, image_bytes)
        with cache_lock:
            cached = detect_cache.get(cache_key)
            if cached:
                print("[DETECT] cache hit")
                payload = dict(cached)
                payload["perf"] = {
                    **payload.get("perf", {}),
                    "cache_hit": True,
                    "total_sec": round(time.time() - request_start, 2),
                }
                return jsonify(payload)

        # Build prompt
        prompt_start = time.time()
        if lang == "fr":
            prompt = (
                "Tu es un assistant visuel utile pour une personne malvoyante. "
                "Réponds à la question de l'utilisateur en te basant sur l'image. "
                "Réponse courte, claire, naturelle et honnête. "
                "Si l'image ne permet pas de répondre clairement, dis-le. "
                f"Question: {question}"
            )
        else:
            prompt = (
                "You are a helpful visual assistant for a visually impaired person. "
                "Answer the user's question based on the image. "
                "Keep the answer short, clear, natural, and honest. "
                "If the image does not allow a clear answer, say so. "
                f"Question: {question}"
            )

        # Inference
        model_start = time.time()
        answer = call_ollama_vision(image_bytes, prompt)
        model_time = perf("DETECT - Qwen3-VL inference", model_start)

        if not answer:
            answer = "Je ne peux pas répondre clairement à partir de cette image." if lang == "fr" else "I cannot answer clearly from this image."

        total_time = perf("DETECT - total request", request_start)

        print(f"[DETECT-{lang}] Q: {question}")
        print(f"[DETECT-{lang}] A: {answer}")

        response_payload = {
            "question": question,
            "answer": answer,
            "lang": lang,
            "perf": {
                "model_sec": round(model_time, 2),
                "total_sec": round(total_time, 2),
                "cache_hit": False,
            }
        }

        # FIX: Thread-safe Cache Setting
        with cache_lock:
            detect_cache[cache_key] = response_payload

        return jsonify(response_payload)

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Ollama is not running. Start it first with: ollama serve", "retryable": True}), 503
    except Exception as e:
        print(f"[DETECT ERROR] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/tts", methods=["POST"])
def tts():
    """Text-to-Speech using local Piper - FIX: In-Memory I/O"""
    request_start = time.time()

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body"}), 400

        text = data.get("text", "").strip()
        lang = data.get("lang", "en").strip().lower()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Lazy load the requested voice
        voice = get_piper_voice(lang)
        if not voice:
            return jsonify({"error": f"Language '{lang}' not supported or model missing"}), 400

        synth_start = time.time()
        
        # FIX: Generate audio entirely in RAM instead of writing to disk
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
            
        audio_buffer.seek(0) # Rewind the buffer to the beginning for reading
        
        synth_time = perf("TTS - Piper synthesis", synth_start)
        total_time = perf("TTS - total request", request_start)

        print(f"[TTS-{lang}] {text}")

        # Send the file out of RAM directly
        response = send_file(audio_buffer, mimetype="audio/wav", as_attachment=False, download_name="response.wav")
        response.headers["X-TTS-Total-Sec"] = f"{total_time:.2f}"
        response.headers["X-TTS-Synthesis-Sec"] = f"{synth_time:.2f}"
        return response

    except Exception as e:
        print(f"[TTS ERROR] {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=== Server ready ===")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)