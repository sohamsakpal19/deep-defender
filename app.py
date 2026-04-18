import ipaddress
import os
import random
import socket
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from html.parser import HTMLParser

from flask import Flask, abort, jsonify, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_FILES = {"index.html", "script.js", "style.css"}
MAX_HTML_BYTES = 1_000_000
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
DEFAULT_SQLITE_PATH = os.getenv("SQLITE_PATH", os.path.join(BASE_DIR, "detections.db"))

SUSPICIOUS_TERMS = {
    "deepfake", "face-swap", "faceswap", "synthetic", "ai-generated",
    "ai_voice", "voice-clone", "voiceclone", "cloned", "fake", "forged"
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# ─── DATABASE CONFIG ──────────────────────────────────────────────
database_url = os.getenv("DATABASE_URL", "")
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

if not database_url:
    # Local development falls back to SQLite automatically. On Render, you can
    # either provide DATABASE_URL for Postgres or point SQLITE_PATH at a disk mount.
    database_url = f"sqlite:///{DEFAULT_SQLITE_PATH}"

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SEED_DEMO_DATA"] = os.getenv("SEED_DEMO_DATA", "true").lower() == "true"

db = SQLAlchemy(app)


# ─── MODELS ────────────────────────────────────────────────────────
class ImageDetection(db.Model):
    __tablename__ = "image_detections"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.Text, nullable=False)
    result = db.Column(db.Text, nullable=False)  # 'FAKE' or 'REAL'
    confidence = db.Column(db.Float, nullable=False)  # 0.0 to 1.0
    model_used = db.Column(db.Text, default="ResNet-50")
    detected_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "result": self.result,
            "confidence": self.confidence,
            "model_used": self.model_used,
            "detected_at": self.detected_at.strftime("%Y-%m-%d %H:%M:%S"),
        }


class VoiceDetection(db.Model):
    __tablename__ = "voice_detections"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.Text, nullable=False)
    result = db.Column(db.Text, nullable=False)  # 'FAKE' or 'REAL'
    confidence = db.Column(db.Float, nullable=False)
    duration_sec = db.Column(db.Float, default=0.0)
    model_used = db.Column(db.Text, default="WaveNet-Detector")
    detected_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "result": self.result,
            "confidence": self.confidence,
            "duration_sec": self.duration_sec,
            "model_used": self.model_used,
            "detected_at": self.detected_at.strftime("%Y-%m-%d %H:%M:%S"),
        }


class UrlDetection(db.Model):
    __tablename__ = "url_detections"

    id = db.Column(db.Integer, primary_key=True)
    source_url = db.Column(db.Text, nullable=False)
    resolved_url = db.Column(db.Text, nullable=False)
    media_type = db.Column(db.Text, nullable=False)
    result = db.Column(db.Text, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    model_used = db.Column(db.Text, nullable=False)
    reason = db.Column(db.Text, nullable=False)
    detected_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "source_url": self.source_url,
            "resolved_url": self.resolved_url,
            "media_type": self.media_type,
            "result": self.result,
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "model_used": self.model_used,
            "reason": self.reason,
            "detected_at": self.detected_at.strftime("%Y-%m-%d %H:%M:%S"),
        }


# ─── URL / MEDIA ANALYSIS HELPERS ──────────────────────────────────
class MediaLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.candidates = []

    def handle_starttag(self, tag, attrs):
        attr_map = dict(attrs)
        if tag == "meta":
            key = attr_map.get("property") or attr_map.get("name")
            if key in {"og:image", "og:audio", "og:video", "twitter:image"}:
                content = attr_map.get("content")
                if content:
                    self.candidates.append(content)
            return

        for attr in ("src", "href"):
            value = attr_map.get(attr)
            if value:
                self.candidates.append(value)


def classify_media_type(content_type, url):
    content_type = (content_type or "").split(";")[0].strip().lower()
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("audio/"):
        return "audio"

    path = urllib.parse.urlparse(url).path.lower()
    _, ext = os.path.splitext(path)
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return None


def score_keywords(*values):
    text = " ".join(v for v in values if v).lower()
    hits = sorted(term for term in SUSPICIOUS_TERMS if term in text)
    score = min(len(hits) * 0.14, 0.42)
    reasons = [f"URL metadata contains suspicious term '{term}'" for term in hits[:3]]
    return score, reasons


def is_private_hostname(hostname):
    if not hostname:
        return True
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False

    for info in infos:
        ip_text = info[4][0]
        try:
            ip_obj = ipaddress.ip_address(ip_text)
        except ValueError:
            continue
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
        ):
            return True
    return False


def validate_user_url(raw_url):
    parsed = urllib.parse.urlparse((raw_url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are supported.")
    if not parsed.netloc:
        raise ValueError("Please enter a complete URL.")
    if is_private_hostname(parsed.hostname):
        raise ValueError("Private, local, or reserved hosts are not allowed.")
    return parsed.geturl()


def build_request(url):
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": "DeepDefender/1.0 (+url-analysis)",
            "Accept": "*/*",
        },
    )


def read_limited(response, limit):
    data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("The remote file is too large to analyze.")
    return data


def choose_media_candidate(html_text, base_url):
    parser = MediaLinkParser()
    parser.feed(html_text)
    for candidate in parser.candidates:
        absolute = urllib.parse.urljoin(base_url, candidate)
        media_type = classify_media_type("", absolute)
        if media_type:
            return absolute
    return None


def fetch_remote_media(url, depth=0):
    if depth > 1:
        raise ValueError("Could not resolve a supported media file from the URL.")

    with urllib.request.urlopen(build_request(url), timeout=20) as response:
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        media_type = classify_media_type(content_type, final_url)

        if media_type is None and "text/html" in content_type.lower():
            html_bytes = read_limited(response, MAX_HTML_BYTES)
            html_text = html_bytes.decode("utf-8", errors="ignore")
            candidate = choose_media_candidate(html_text, final_url)
            if not candidate:
                raise ValueError("No supported image or audio file was found at that URL.")
            return fetch_remote_media(candidate, depth + 1)

        if media_type is None:
            raise ValueError("The URL does not point to a supported image or audio file.")

        path = urllib.parse.urlparse(final_url).path
        _, ext = os.path.splitext(path)
        suffix = ext if ext else (".jpg" if media_type == "image" else ".wav")
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=BASE_DIR)

        try:
            total = 0
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError("The remote file is too large to analyze.")
                tmp_file.write(chunk)
        finally:
            tmp_file.close()

        return {
            "file_path": tmp_file.name,
            "source_url": url,
            "resolved_url": final_url,
            "content_type": content_type,
            "media_type": media_type,
        }


def try_load_image_modules():
    try:
        import cv2
        import numpy as np
        from image_preprocess import load_and_preprocess_image
        return {"cv2": cv2, "np": np, "preprocess": load_and_preprocess_image}
    except Exception:
        return None


def try_load_audio_modules():
    try:
        import librosa
        import numpy as np
        from audio_preprocess import load_and_preprocess_audio
        return {"librosa": librosa, "np": np, "preprocess": load_and_preprocess_audio}
    except Exception:
        return None


def load_optional_model(media_type):
    if media_type == "image":
        weight_candidates = [
            os.path.join(BASE_DIR, "models", "image_xception.weights.h5"),
            os.path.join(BASE_DIR, "image_xception.weights.h5"),
        ]
        try:
            import tensorflow as tf  # noqa: F401
            from image_xception import build_xception_classifier, load_xception_weights

            for candidate in weight_candidates:
                if os.path.exists(candidate):
                    model = build_xception_classifier()
                    return load_xception_weights(model, candidate), candidate
        except Exception:
            return None, None

    if media_type == "audio":
        weight_candidates = [
            os.path.join(BASE_DIR, "models", "audio_transformer.weights.h5"),
            os.path.join(BASE_DIR, "audio_transformer.weights.h5"),
        ]
        try:
            import tensorflow as tf  # noqa: F401
            from audio_transformer import build_audio_transformer

            for candidate in weight_candidates:
                if os.path.exists(candidate):
                    model = build_audio_transformer()
                    model.load_weights(candidate)
                    return model, candidate
        except Exception:
            return None, None

    return None, None


def heuristic_image_analysis(file_path, source_url, resolved_url):
    modules = try_load_image_modules()
    score, reasons = score_keywords(source_url, resolved_url)

    if not modules:
        final_score = min(max(score + 0.18, 0.08), 0.92)
        return final_score, reasons or ["Image modules unavailable; using URL metadata fallback."], "metadata-heuristic"

    cv2 = modules["cv2"]
    np = modules["np"]
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError("Downloaded image could not be decoded.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
    contrast = float((gray.max() - gray.min()) / 255.0)

    if laplacian_var < 45:
        score += 0.22
        reasons.append("Image appears unusually smooth with low edge detail.")
    if saturation > 0.68:
        score += 0.12
        reasons.append("Image shows unusually strong color saturation.")
    if contrast < 0.22:
        score += 0.10
        reasons.append("Image contrast is compressed in a way often seen after synthetic rendering.")

    return min(score, 0.96), reasons or ["No strong synthetic indicators were found."], "image-heuristic"


def heuristic_audio_analysis(file_path, source_url, resolved_url):
    modules = try_load_audio_modules()
    score, reasons = score_keywords(source_url, resolved_url)

    if not modules:
        final_score = min(max(score + 0.18, 0.08), 0.92)
        return final_score, reasons or ["Audio modules unavailable; using URL metadata fallback."], "metadata-heuristic"

    librosa = modules["librosa"]
    np = modules["np"]
    y, sr = librosa.load(file_path, sr=16000)
    duration = len(y) / sr if sr else 0.0

    if duration <= 0:
        raise ValueError("Downloaded audio could not be decoded.")

    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    rms_var = float(np.std(librosa.feature.rms(y=y)))

    if flatness > 0.18:
        score += 0.20
        reasons.append("Audio has flatter spectral texture than typical natural speech.")
    if rms_var < 0.02:
        score += 0.16
        reasons.append("Audio volume envelope is unusually even across the sample.")
    if duration < 1.5:
        score += 0.08
        reasons.append("Very short clips are harder to verify and more likely to be synthetic samples.")

    return min(score, 0.96), reasons or ["No strong synthetic indicators were found."], "audio-heuristic"


def analyze_media(file_path, media_type, source_url, resolved_url):
    model, weight_path = load_optional_model(media_type)
    if model is not None:
        try:
            if media_type == "image":
                processed = try_load_image_modules()["preprocess"](file_path)
            else:
                processed = try_load_audio_modules()["preprocess"](file_path)
            prediction = float(model.predict(processed, verbose=0)[0][0])
            reasons = [f"Prediction produced by trained {media_type} model."]
            return prediction, reasons, os.path.basename(weight_path)
        except Exception:
            pass

    if media_type == "image":
        return heuristic_image_analysis(file_path, source_url, resolved_url)
    return heuristic_audio_analysis(file_path, source_url, resolved_url)


def make_detection_payload(source_url, media_type, score, reasons, model_used, resolved_url):
    score = max(0.0, min(score, 1.0))
    result = "FAKE" if score >= 0.5 else "REAL"
    confidence = round(score if result == "FAKE" else 1 - score, 2)
    reason = reasons[0] if reasons else "Analysis completed."
    return {
        "source_url": source_url,
        "resolved_url": resolved_url,
        "media_type": media_type,
        "result": result,
        "confidence": confidence,
        "risk_score": round(score, 2),
        "reason": reason,
        "signals": reasons[:3],
        "model_used": model_used,
        "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_url_detection(payload):
    rec = UrlDetection(
        source_url=payload["source_url"],
        resolved_url=payload["resolved_url"],
        media_type=payload["media_type"],
        result=payload["result"],
        confidence=payload["confidence"],
        risk_score=payload["risk_score"],
        model_used=payload["model_used"],
        reason=payload["reason"],
        detected_at=datetime.strptime(payload["detected_at"], "%Y-%m-%d %H:%M:%S"),
    )
    db.session.add(rec)
    db.session.commit()


# ─── DB INIT + DEMO SEED ──────────────────────────────────────────
def init_db():
    db.create_all()


def seed_demo_data():
    if not app.config["SEED_DEMO_DATA"]:
        return

    if ImageDetection.query.count() > 0:
        return

    image_files = [
        "face_001.jpg", "photo_scan.png", "user_upload.jpg", "synthetic_gen.png",
        "portrait_ai.jpg", "news_img.jpg", "social_post.jpg", "id_scan.jpg"
    ]
    voice_files = [
        "audio_clip.wav", "voice_msg.mp3", "call_record.wav", "synthetic_voice.wav",
        "deepfake_audio.mp3", "real_speech.wav", "podcast_seg.mp3", "phone_call.wav"
    ]

    base = datetime.now() - timedelta(days=7)

    for i, f in enumerate(image_files):
        result = "FAKE" if i % 3 != 0 else "REAL"
        conf = round(random.uniform(0.72, 0.99), 2)
        ts = base + timedelta(hours=i * 8)
        db.session.add(
            ImageDetection(filename=f, result=result, confidence=conf, detected_at=ts)
        )

    for i, f in enumerate(voice_files):
        result = "FAKE" if i % 4 != 0 else "REAL"
        conf = round(random.uniform(0.68, 0.98), 2)
        dur = round(random.uniform(2.0, 45.0), 1)
        ts = base + timedelta(hours=i * 9 + 2)
        db.session.add(
            VoiceDetection(
                filename=f,
                result=result,
                confidence=conf,
                duration_sec=dur,
                detected_at=ts,
            )
        )

    db.session.commit()


# ─── IMAGE ROUTES ─────────────────────────────────────────────────
@app.route("/api/image/detect", methods=["POST"])
def image_detect():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "unknown.jpg")
    result = random.choice(["FAKE", "FAKE", "REAL"])
    confidence = round(random.uniform(0.75, 0.99), 2)
    detected_at = datetime.now()

    rec = ImageDetection(
        filename=filename,
        result=result,
        confidence=confidence,
        detected_at=detected_at
    )
    db.session.add(rec)
    db.session.commit()

    return jsonify({
        "result": result,
        "confidence": confidence,
        "filename": filename,
        "detected_at": detected_at.strftime("%Y-%m-%d %H:%M:%S")
    }), 201


@app.route("/api/image/history", methods=["GET"])
def image_history():
    rows = ImageDetection.query.order_by(ImageDetection.detected_at.desc()).limit(20).all()
    return jsonify([r.to_dict() for r in rows])


# ─── VOICE ROUTES ─────────────────────────────────────────��───────
@app.route("/api/voice/detect", methods=["POST"])
def voice_detect():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "unknown.wav")
    duration = round(random.uniform(3.0, 30.0), 1)
    result = random.choice(["FAKE", "FAKE", "REAL"])
    confidence = round(random.uniform(0.70, 0.97), 2)
    detected_at = datetime.now()

    rec = VoiceDetection(
        filename=filename,
        result=result,
        confidence=confidence,
        duration_sec=duration,
        detected_at=detected_at
    )
    db.session.add(rec)
    db.session.commit()

    return jsonify({
        "result": result,
        "confidence": confidence,
        "filename": filename,
        "duration_sec": duration,
        "detected_at": detected_at.strftime("%Y-%m-%d %H:%M:%S")
    }), 201


@app.route("/api/voice/history", methods=["GET"])
def voice_history():
    rows = VoiceDetection.query.order_by(VoiceDetection.detected_at.desc()).limit(20).all()
    return jsonify([r.to_dict() for r in rows])


# ─── URL ANALYSIS ROUTES ──────────────────────────────────────────
@app.route("/api/url/analyze", methods=["POST"])
def url_analyze():
    data = request.get_json(silent=True) or {}
    raw_url = data.get("url", "")

    try:
        source_url = validate_user_url(raw_url)
        remote = fetch_remote_media(source_url)
        try:
            score, reasons, model_used = analyze_media(
                remote["file_path"],
                remote["media_type"],
                source_url,
                remote["resolved_url"],
            )
        finally:
            if os.path.exists(remote["file_path"]):
                os.remove(remote["file_path"])

        payload = make_detection_payload(
            source_url=source_url,
            media_type=remote["media_type"],
            score=score,
            reasons=reasons,
            model_used=model_used,
            resolved_url=remote["resolved_url"],
        )
        save_url_detection(payload)
        return jsonify(payload), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except urllib.error.URLError as exc:
        return jsonify({"error": f"Failed to fetch URL: {exc.reason}"}), 502
    except Exception as exc:
        return jsonify({"error": f"URL analysis failed: {exc}"}), 500


@app.route("/api/url/history", methods=["GET"])
def url_history():
    rows = UrlDetection.query.order_by(UrlDetection.detected_at.desc()).limit(20).all()
    return jsonify([r.to_dict() for r in rows])


# ─── ANALYTICS ────────────────────────────────────────────────────
@app.route("/api/analytics", methods=["GET"])
def analytics():
    def stats(model):
        total = db.session.query(func.count(model.id)).scalar() or 0
        fake = db.session.query(func.count(model.id)).filter(model.result == "FAKE").scalar() or 0
        real = total - fake
        avg_conf = db.session.query(func.avg(model.confidence)).scalar() or 0.0
        return {
            "total": total,
            "fake": fake,
            "real": real,
            "fake_pct": round(fake / total * 100, 1) if total else 0,
            "avg_confidence": round(avg_conf * 100, 1) if avg_conf else 0
        }

    last_7_days = datetime.now() - timedelta(days=7)

    daily_img_rows = (
        db.session.query(
            func.date(ImageDetection.detected_at).label("day"),
            ImageDetection.result.label("result"),
            func.count(ImageDetection.id).label("cnt"),
        )
        .filter(ImageDetection.detected_at >= last_7_days)
        .group_by(func.date(ImageDetection.detected_at), ImageDetection.result)
        .order_by(func.date(ImageDetection.detected_at))
        .all()
    )

    daily_voice_rows = (
        db.session.query(
            func.date(VoiceDetection.detected_at).label("day"),
            VoiceDetection.result.label("result"),
            func.count(VoiceDetection.id).label("cnt"),
        )
        .filter(VoiceDetection.detected_at >= last_7_days)
        .group_by(func.date(VoiceDetection.detected_at), VoiceDetection.result)
        .order_by(func.date(VoiceDetection.detected_at))
        .all()
    )

    daily_image = [
        {"day": str(r.day), "result": r.result, "cnt": int(r.cnt)}
        for r in daily_img_rows
    ]
    daily_voice = [
        {"day": str(r.day), "result": r.result, "cnt": int(r.cnt)}
        for r in daily_voice_rows
    ]

    return jsonify({
        "image": stats(ImageDetection),
        "voice": stats(VoiceDetection),
        "daily_image": daily_image,
        "daily_voice": daily_voice
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ─── FRONTEND SERVING ─────────────────────────────────────────────
@app.route("/")
@app.route("/index.html")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_file(filename):
    if filename.startswith("api/") or filename not in FRONTEND_FILES:
        abort(404)
    return send_from_directory(BASE_DIR, filename)


@app.after_request
def disable_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def bootstrap_app():
    with app.app_context():
        init_db()
        seed_demo_data()


bootstrap_app()


def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9000"))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    print(f"DeepDetect running -> http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
