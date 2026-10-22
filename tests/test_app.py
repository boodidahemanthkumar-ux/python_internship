import io
import os
import shutil
import pytest
from PIL import Image

import config
from app import create_app


def create_test_image_bytes(format="JPEG", size=(100, 100), color=(73, 109, 137)):
    """Generate in-memory valid image bytes for testing."""
    image = Image.new("RGB", size, color=color)
    byte_arr = io.BytesIO()
    image.save(byte_arr, format=format)
    byte_arr.seek(0)
    return byte_arr


class MockSuccessDetector:
    """Mock detector returning valid detections for testing without neural net overhead."""

    def __init__(self):
        self.called = False

    def detect_and_draw(self, image_path, output_path, conf_threshold=0.25):
        self.called = True
        # Simulate creating the output image
        shutil.copyfile(image_path, output_path)
        return {
            "success": True,
            "total_detections": 2,
            "detections": [
                {
                    "id": 1,
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.942,
                    "confidence_pct": 94.2,
                    "bbox": [10, 20, 80, 120],
                    "color": "#2563eb",
                },
                {
                    "id": 2,
                    "class_id": 2,
                    "class_name": "car",
                    "confidence": 0.885,
                    "confidence_pct": 88.5,
                    "bbox": [90, 40, 180, 140],
                    "color": "#10b981",
                },
            ],
            "class_counts": {"person": 1, "car": 1},
            "processing_time": 0.0385,
            "processing_time_ms": 38.5,
            "image_dimensions": {"width": 200, "height": 200},
            "conf_threshold": conf_threshold,
        }


class MockFailingDetector:
    """Mock detector simulating runtime inference crash."""

    def detect_and_draw(self, image_path, output_path, conf_threshold=0.25):
        raise RuntimeError("Simulated inference engine exception")


@pytest.fixture
def client():
    """Test client fixture with MockSuccessDetector."""
    mock_detector = MockSuccessDetector()
    app = create_app(detector_instance=mock_detector)
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def failing_client():
    """Test client fixture with MockFailingDetector."""
    mock_detector = MockFailingDetector()
    app = create_app(detector_instance=mock_detector)
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as test_client:
        yield test_client


def test_home_page_loads(client):
    """Verify that the home page (index) renders successfully with 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Image Recognition & Object Detection" in html
    assert "YOLO11n" in html
    assert "dropzone" in html


def test_health_check_endpoint(client):
    """Verify that the /api/health endpoint returns 200 OK with healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "healthy"
    assert json_data["model"] == "YOLO11n"


def test_no_file_rejected(client):
    """Verify that submitting without a file redirects back to home with an error."""
    response = client.post("/detect", data={}, follow_redirects=True)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "No file was uploaded" in html


def test_empty_filename_rejected(client):
    """Verify that submitting an empty filename is rejected."""
    data = {"image": (io.BytesIO(b""), "")}
    response = client.post(
        "/detect", data=data, content_type="multipart/form-data", follow_redirects=True
    )
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "No file was selected" in html


def test_unsupported_file_rejected(client):
    """Verify that submitting an unsupported file format (e.g. .txt) is rejected."""
    data = {"image": (io.BytesIO(b"This is a text file"), "notes.txt")}
    response = client.post(
        "/detect", data=data, content_type="multipart/form-data", follow_redirects=True
    )
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Unsupported file format" in html


def test_corrupted_image_rejected(client):
    """Verify that corrupted image bytes fail OpenCV decoding and are rejected."""
    corrupt_bytes = io.BytesIO(b"NOT_A_REAL_IMAGE_BYTES_XYZ123456789")
    data = {"image": (corrupt_bytes, "broken.jpg")}
    response = client.post(
        "/detect", data=data, content_type="multipart/form-data", follow_redirects=True
    )
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "corrupted or not a valid image" in html


def test_oversized_upload_rejected(client):
    """Verify that uploads exceeding MAX_CONTENT_LENGTH (10 MB) are rejected."""
    oversized_size = config.MAX_CONTENT_LENGTH + 1024  # Just above 10MB
    oversized_bytes = io.BytesIO(b"X" * oversized_size)
    data = {"image": (oversized_bytes, "massive.png")}

    response = client.post(
        "/detect", data=data, content_type="multipart/form-data", follow_redirects=True
    )
    # Flask MAX_CONTENT_LENGTH triggers 413 which our errorhandler redirects to index
    assert response.status_code in [200, 413]
    html = response.data.decode("utf-8")
    assert "exceeds" in html or "limit" in html or "413" in html


def test_successful_detection(client):
    """Verify that a valid image upload processes successfully and renders detection analytics."""
    img_bytes = create_test_image_bytes(format="JPEG")
    data = {
        "image": (img_bytes, "sample.jpg"),
        "confidence": "0.30",
    }
    response = client.post(
        "/detect", data=data, content_type="multipart/form-data"
    )
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Detection Analytics" in html
    assert "person" in html
    assert "car" in html
    assert "94.2%" in html
    assert "Visual Comparison" in html


def test_detection_failure_does_not_crash(failing_client):
    """Verify that an inference failure in detector does not crash the server and shows an error."""
    img_bytes = create_test_image_bytes(format="PNG")
    data = {
        "image": (img_bytes, "test.png"),
        "confidence": "0.25",
    }
    response = failing_client.post(
        "/detect", data=data, content_type="multipart/form-data", follow_redirects=True
    )
    # Server should catch exception, log it, flash error, and redirect gracefully
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "An error occurred while processing the image" in html


def test_download_missing_file_returns_404(client):
    """Verify that attempting to download a nonexistent result file returns 404."""
    response = client.get("/download/nonexistent_file_99999.jpg")
    assert response.status_code == 404
