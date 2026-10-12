import os
import sys
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from app import app

    @app.route("/api/debug")
    def debug():
        return {
            "status": "ok",
            "cwd": os.getcwd(),
            "base_dir": BASE_DIR,
            "root_files": os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else [],
            "has_artifacts": os.path.exists(os.path.join(BASE_DIR, "artifacts")),
            "has_templates": os.path.exists(os.path.join(BASE_DIR, "templates")),
        }

except Exception as e:
    tb = traceback.format_exc()
    from flask import Flask
    app = Flask(__name__)

    @app.route("/api/debug")
    @app.route("/")
    def debug_err():
        return f"<h1>Startup Exception on Vercel:</h1><pre>{tb}</pre>", 200
