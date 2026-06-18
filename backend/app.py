"""
Flask API — X光肺炎辅助诊断系统
"""
import os, sys, uuid
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

sys.path.append(os.path.dirname(__file__))
from predict import get_predictor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["ALLOWED_EXTENSIONS"] = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
CORS(app)


def allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in app.config["ALLOWED_EXTENSIONS"]


def save_upload(file):
    ext = os.path.splitext(file.filename)[1].lower()
    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], name)
    file.save(path)
    return path


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "未上传文件"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "文件名为空"}), 400
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "不支持的文件格式"}), 400

    try:
        file_path = save_upload(file)
        img = Image.open(file_path).convert("RGB")
        predictor = get_predictor()
        result = predictor.predict(img)
        os.remove(file_path)
        return jsonify({"success": True, **result})
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": f"模型未找到: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": f"预测失败: {str(e)}"}), 500


@app.route("/api/class_info", methods=["GET"])
def class_info():
    from predict import CLASS_NAMES, CLASS_COLORS, CLASS_ADVICE
    classes = [{"name": n, "color": CLASS_COLORS.get(n), "advice": CLASS_ADVICE.get(n)} for n in CLASS_NAMES]
    return jsonify({"classes": classes})


@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False, "error": "文件过大，最大16MB"}), 413


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_path", default=None)
    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════╗
║  肺炎X光辅助诊断系统 — API 服务           ║
║  ResNet50 + CBAM 注意力机制               ║
╚═══════════════════════════════════════════╝
    http://{args.host}:{args.port}
    POST /api/predict    — 上传图像预测
    GET  /api/health     — 健康检查
    GET  /api/class_info — 分类信息
    """)
    try:
        get_predictor(args.model_path)
    except Exception as e:
        print(f"⚠️ {e}")

    app.run(host=args.host, port=args.port, debug=args.debug)
