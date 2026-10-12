"""
Used Car Price Estimator - Flask backend with AI Buying Advisory & Image Inspection

    python app.py          ->  http://127.0.0.1:5000

REST API
    GET    /api/health            model + storage status
    GET    /api/meta              brands, models, fuels, inspection checkpoints
    POST   /api/analyze-images    upload car photos -> visual condition assessment & ratings
    POST   /api/predict           JSON in -> valuation, advisory, deal sheet, negotiation
    GET    /api/insights          training metrics, loss curves, feature importance
    GET    /api/history           recent valuations & deal sheets
    GET    /api/history/<id>      retrieve single valuation report
    DELETE /api/history/<id>      remove one
    DELETE /api/history           remove all
"""
import base64
import os

from flask import Flask, jsonify, render_template, request, send_from_directory

from db import Store
from ml.image_analyzer import aggregate_images_analysis, analyze_car_image
from ml.predictor import ModelNotTrained, Predictor, ValidationError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
# Allow max 32MB file uploads for multi-angle car photos
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
CORS(app)

db_path = "/tmp/history.json" if (os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME")) else os.path.join(BASE_DIR, "data", "history.json")
store = Store(file_path=db_path)
try:
    predictor = Predictor(artifacts_dir=os.path.join(BASE_DIR, "artifacts"))
    load_error = None
except ModelNotTrained as e:
    predictor, load_error = None, str(e)


def _need_model():
    if predictor is None:
        return jsonify(error=load_error, field=None), 503
    return None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)


@app.get("/api/health")
def health():
    m = predictor.metrics if predictor else {}
    res = m.get("results", {}).get("Deep neural network", {})
    return jsonify(
        model_ready=predictor is not None,
        message=load_error,
        storage=store.backend,
        predictions=store.count(),
        params=m.get("params"),
        r2=res.get("r2"),
    )


@app.get("/api/meta")
def meta():
    bad = _need_model()
    return bad or jsonify(predictor.meta())


@app.post("/api/analyze-images")
def analyze_images_endpoint():
    results = []
    # 1. Handle multipart form-data
    if request.files:
        files = request.files.getlist("images")
        for f in files:
            content = f.read()
            if content:
                res = analyze_car_image(content, filename=f.filename)
                results.append(res)
    # 2. Handle JSON base64 payloads
    elif request.is_json:
        payload = request.get_json(silent=True) or {}
        images_data = payload.get("images", [])
        for item in images_data:
            data_url = item.get("data", "")
            name = item.get("name", "car.jpg")
            if "," in data_url:
                data_url = data_url.split(",", 1)[1]
            try:
                raw_bytes = base64.b64decode(data_url)
                res = analyze_car_image(raw_bytes, filename=name)
                results.append(res)
            except Exception as e:
                results.append({"error": str(e), "filename": name})

    analysis = aggregate_images_analysis(results)
    return jsonify(analysis)


@app.post("/api/predict")
def predict():
    bad = _need_model()
    if bad:
        return bad
    payload = request.get_json(silent=True) or {}
    try:
        result = predictor.predict(payload)
    except ValidationError as e:
        return jsonify(error=e.message, field=e.field), 400

    car = result["car"]
    advisory = result.get("advisory", {})
    images_meta = advisory.get("image_analysis_summary", {})
    first_thumb = None
    if images_meta.get("gallery"):
        first_thumb = images_meta["gallery"][0].get("thumbnail")

    record_id = store.add({
        "brand": car["brand"], "model": car["model"], "year": car["year"], "km": car["km"],
        "fuel": car["fuel"], "transmission": car["transmission"], "owner": car["owner"],
        "price": result["price"], "low": result["low"], "high": result["high"],
        "asking": result["deal"]["asking"] if result["deal"] else None,
        "verdict_badge": advisory.get("verdict_badge"),
        "overall_buy_index": advisory.get("overall_buy_index"),
        "verdict_color": advisory.get("verdict_color"),
        "thumbnail": first_thumb,
        "defects_count": len(advisory.get("deal_sheet", {}).get("identified_defects", [])),
        "refurbishment_cost": advisory.get("deal_sheet", {}).get("refurbishment_cost", 0),
        "full_result": result,
    })
    result["id"] = record_id
    return jsonify(result)


@app.get("/api/insights")
def insights():
    bad = _need_model()
    return bad or jsonify(predictor.metrics)


@app.get("/api/history")
def history():
    limit = min(int(request.args.get("limit", 50)), 200)
    return jsonify(items=store.list(limit), total=store.count(), storage=store.backend)


@app.get("/api/history/<rec_id>")
def history_detail(rec_id):
    item = store.get(rec_id)
    if not item:
        return jsonify(error="Record not found"), 404
    return jsonify(item)


@app.delete("/api/history/<rec_id>")
def history_delete(rec_id):
    return jsonify(deleted=store.delete(rec_id))


@app.delete("/api/history")
def history_clear():
    store.clear()
    return jsonify(deleted=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=False)
