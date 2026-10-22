import os
import uuid
import logging
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    jsonify,
    abort,
)
import cv2

import config
from detector import ObjectDetector, load_image_safely

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has a permitted extension."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in config.ALLOWED_EXTENSIONS


def create_app(detector_instance=None) -> Flask:
    """
    Application factory for the Image Recognition and Object Detection System.
    Accepts an optional pre-instantiated or mocked detector for modularity and testing.
    """
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
    app.config["RESULT_FOLDER"] = config.RESULT_FOLDER

    # Ensure runtime directories exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

    # Initialize YOLO detector once at startup if not provided
    if detector_instance is not None:
        app.detector = detector_instance
    else:
        logger.info("Initializing global YOLO model on server startup...")
        app.detector = ObjectDetector(model_name=config.MODEL_PATH)

    @app.route("/", methods=["GET"])
    def index():
        """Render the upload and detection configuration page."""
        return render_template(
            "index.html",
            default_conf=config.DEFAULT_CONF_THRESHOLD,
            min_conf=config.MIN_CONF_THRESHOLD,
            max_conf=config.MAX_CONF_THRESHOLD,
        )

    @app.route("/detect", methods=["POST"])
    def detect():
        """
        Handle image upload, input validation, object detection inference,
        and rendering of detection results.
        """
        # 1. Check if the image part is in the request
        if "image" not in request.files:
            flash("No file was uploaded. Please select an image.", "error")
            return redirect(url_for("index"))

        file = request.files["image"]

        # 2. Check for empty filename
        if not file or file.filename.strip() == "":
            flash("No file was selected. Please choose a valid image file.", "error")
            return redirect(url_for("index"))

        # 3. Validate file extension
        if not allowed_file(file.filename):
            allowed_list = ", ".join(sorted(ext.upper() for ext in config.ALLOWED_EXTENSIONS))
            flash(
                f"Unsupported file format. Please upload one of: {allowed_list}.",
                "error",
            )
            return redirect(url_for("index"))

        # 4. Parse confidence threshold
        conf_raw = request.form.get("confidence", str(config.DEFAULT_CONF_THRESHOLD))
        try:
            conf_threshold = float(conf_raw)
            conf_threshold = max(config.MIN_CONF_THRESHOLD, min(config.MAX_CONF_THRESHOLD, conf_threshold))
        except (ValueError, TypeError):
            conf_threshold = config.DEFAULT_CONF_THRESHOLD

        # 5. Sanitize and generate unique filename with UUID (never trust user input)
        file_ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{file_ext}"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        result_path = os.path.join(app.config["RESULT_FOLDER"], unique_name)

        try:
            file.save(upload_path)
        except Exception as exc:
            logger.error(f"Failed to save uploaded file: {exc}")
            flash("Failed to save the uploaded image. Please try again.", "error")
            return redirect(url_for("index"))

        # 6. Validate image integrity by decoding with OpenCV
        test_img = load_image_safely(upload_path)
        if test_img is None:
            # File is corrupt or not a true image
            if os.path.exists(upload_path):
                os.remove(upload_path)
            flash(
                "Uploaded file appears to be corrupted or not a valid image.",
                "error",
            )
            return redirect(url_for("index"))

        # 7. Run Object Detection via detector
        try:
            results_data = app.detector.detect_and_draw(
                image_path=upload_path,
                output_path=result_path,
                conf_threshold=conf_threshold,
            )
        except Exception as exc:
            logger.error(f"Inference error during detection: {exc}", exc_info=True)
            if os.path.exists(upload_path):
                os.remove(upload_path)
            if os.path.exists(result_path):
                os.remove(result_path)
            flash("An error occurred while processing the image. Please try another image.", "error")
            return redirect(url_for("index"))

        # 8. Render detection result page
        return render_template(
            "result.html",
            filename=unique_name,
            original_url=url_for("serve_upload", filename=unique_name),
            result_url=url_for("serve_result", filename=unique_name),
            data=results_data,
        )

    @app.route("/uploads/<filename>")
    def serve_upload(filename: str):
        """Serve original uploaded images securely."""
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

    @app.route("/results/<filename>")
    def serve_result(filename: str):
        """Serve processed result images securely."""
        return send_from_directory(app.config["RESULT_FOLDER"], filename)

    @app.route("/download/<filename>")
    def download_result(filename: str):
        """Download processed image as an attachment."""
        target_path = os.path.join(app.config["RESULT_FOLDER"], filename)
        if not os.path.exists(target_path):
            abort(404)
        return send_from_directory(
            app.config["RESULT_FOLDER"],
            filename,
            as_attachment=True,
            download_name=f"detected_{filename}",
        )

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """API Health check endpoint."""
        return jsonify(
            {
                "status": "healthy",
                "service": "Image Recognition and Object Detection System",
                "model": "YOLO11n",
            }
        )

    # Error Handlers
    @app.errorhandler(413)
    def file_too_large(error):
        max_mb = config.MAX_CONTENT_LENGTH // (1024 * 1024)
        flash(f"File size exceeds the {max_mb} MB limit. Please upload a smaller image.", "error")
        return redirect(url_for("index"))

    @app.errorhandler(404)
    def not_found_error(error):
        return render_template("index.html", error_msg="The requested page or resource was not found (404)."), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return render_template("index.html", error_msg="An internal server error occurred (500). Please try again."), 500

    return app


# Lazy application holder for WSGI servers / flask CLI
_app_instance = None


def __getattr__(name: str):
    """Enable lazy module-level access to 'app' (e.g. flask run, gunicorn)."""
    global _app_instance
    if name == "app":
        if _app_instance is None:
            _app_instance = create_app()
        return _app_instance
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


if __name__ == "__main__":
    server_app = create_app()
    logger.info(f"Starting Image Detection Flask server at http://{config.HOST}:{config.PORT}")
    server_app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
    )

