import os
import time
import logging
import cv2
import numpy as np

# Configure logging for academic and presentation clarity
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Predefined curated color palette (BGR format for OpenCV)
# These colors provide distinct, high-contrast, visually pleasing bounding boxes
PALETTE = [
    (235, 130, 37),   # Bright Blue
    (67, 185, 41),    # Emerald Green
    (38, 70, 237),    # Coral Red
    (196, 56, 172),   # Purple/Violet
    (40, 190, 245),   # Golden Yellow
    (204, 150, 40),   # Cyan
    (80, 80, 240),    # Soft Salmon
    (140, 200, 60),   # Light Mint
    (180, 90, 240),   # Pinkish Lavender
    (50, 140, 230),   # Amber Orange
]


def get_color_for_class(cls_id: int):
    """
    Deterministically assign a consistent, aesthetically pleasing BGR color
    to an object class based on its class ID.
    """
    return PALETTE[cls_id % len(PALETTE)]


def bgr_to_hex(bgr_color):
    """Convert an OpenCV BGR tuple to an HTML Hex color code."""
    b, g, r = bgr_color
    return f"#{r:02x}{g:02x}{b:02x}"


def load_image_safely(image_path: str):
    """
    Safely decode an image file from disk into a BGR numpy array using an
    in-memory byte buffer. Avoids OS-specific path encoding/Unicode bugs in cv2.imread.
    """
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.warning(f"Failed to read image safely from {image_path}: {exc}")
        return None


def save_image_safely(output_path: str, image_bgr: np.ndarray) -> bool:
    """
    Safely encode and write an image to disk via cv2.imencode.
    Avoids OS-specific path encoding/Unicode bugs in cv2.imwrite.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ext = os.path.splitext(output_path)[1]
        if not ext:
            ext = ".jpg"
        success, encoded_img = cv2.imencode(ext, image_bgr)
        if not success:
            return False
        with open(output_path, "wb") as f:
            f.write(encoded_img)
        return True
    except Exception as exc:
        logger.error(f"Failed to save image safely to {output_path}: {exc}")
        return False


class ObjectDetector:
    """
    ObjectDetector handles loading the pretrained YOLO11n model,
    running object detection inference, and annotating images with
    high-quality bounding boxes and metadata.
    """

    def __init__(self, model_name: str = "yolo11n.pt"):
        """
        Initialize the YOLO object detector.
        Ultralytics will automatically download the pretrained weights
        if not already present locally in the workspace.
        """
        # If model_name is an existing path, use it; otherwise, if it's a path pointing to yolo11n.pt that doesn't exist yet, pass "yolo11n.pt" so Ultralytics automatically downloads it
        if os.path.exists(model_name):
            self.model_name = model_name
        else:
            self.model_name = os.path.basename(model_name) if model_name.endswith(".pt") else model_name

        logger.info(f"Loading YOLO model: '{self.model_name}'...")
        logger.info(
            "Note: On first execution, YOLO11n weights (~5.4 MB) will be downloaded "
            "automatically from Ultralytics servers. An active internet connection is required."
        )

        from ultralytics import YOLO

        try:
            self.model = YOLO(self.model_name)
            logger.info(f"Model '{self.model_name}' successfully loaded into memory.")
        except Exception as exc:
            logger.error(f"Failed to load YOLO model: {exc}")
            raise RuntimeError(f"Could not initialize YOLO model: {exc}") from exc

    def detect_and_draw(
        self,
        image_path: str,
        output_path: str,
        conf_threshold: float = 0.25,
    ) -> dict:
        """
        Perform object detection on an image, draw styled bounding boxes,
        and save the annotated image to output_path.

        Parameters:
            image_path (str): Path to input image file.
            output_path (str): Path where the annotated image will be saved.
            conf_threshold (float): Confidence threshold for detections (0.05 to 1.0).

        Returns:
            dict: Structured detection results including class counts, coordinates,
                  confidences, and processing benchmark times.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        # Validate and read the image safely
        image = load_image_safely(image_path)
        if image is None:
            raise ValueError(
                "Image could not be decoded. The file may be corrupted, empty, "
                "or saved in an unsupported format."
            )

        height, width = image.shape[:2]

        # Execute YOLO inference and measure inference time
        start_time = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            verbose=False,
            device="cpu",  # Default to CPU for universal compatibility
        )
        elapsed_time = time.perf_counter() - start_time

        detections = []
        class_counts = {}

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for idx in range(len(boxes)):
                    box = boxes[idx]

                    # Extract coordinates
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]

                    # Clamp coordinates to image boundaries
                    x1 = max(0, min(width - 1, x1))
                    y1 = max(0, min(height - 1, y1))
                    x2 = max(0, min(width - 1, x2))
                    y2 = max(0, min(height - 1, y2))

                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    # Retrieve class label from model metadata
                    cls_name = (
                        self.model.names.get(cls_id, f"Class_{cls_id}")
                        if hasattr(self.model, "names")
                        else f"Class_{cls_id}"
                    )

                    # Update class-wise count
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                    # Aesthetic color coding
                    bgr_color = get_color_for_class(cls_id)
                    hex_color = bgr_to_hex(bgr_color)

                    # Draw bounding box rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), bgr_color, thickness=2, lineType=cv2.LINE_AA)

                    # Label format: "person 94.2%"
                    label = f"{cls_name} {conf * 100:.1f}%"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.52
                    font_thickness = 1

                    (label_w, label_h), baseline = cv2.getTextSize(
                        label, font, font_scale, font_thickness
                    )

                    # Decide label badge position (above box, or inside if too close to top edge)
                    if y1 - label_h - 8 > 0:
                        badge_y1 = y1 - label_h - 8
                        badge_y2 = y1
                        text_y = y1 - 4
                    else:
                        badge_y1 = y1
                        badge_y2 = y1 + label_h + 8
                        text_y = y1 + label_h + 4

                    badge_x1 = x1
                    badge_x2 = min(width, x1 + label_w + 8)

                    # Draw filled background pill/rectangle for the label badge
                    cv2.rectangle(
                        image,
                        (badge_x1, badge_y1),
                        (badge_x2, badge_y2),
                        bgr_color,
                        thickness=-1,
                    )

                    # Draw crisp white text inside the label badge
                    cv2.putText(
                        image,
                        label,
                        (badge_x1 + 4, text_y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        font_thickness,
                        lineType=cv2.LINE_AA,
                    )

                    detections.append(
                        {
                            "id": idx + 1,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": round(conf, 4),
                            "confidence_pct": round(conf * 100, 1),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "color": hex_color,
                        }
                    )

        # Save annotated image safely
        save_success = save_image_safely(output_path, image)
        if not save_success:
            raise RuntimeError(f"Failed to write annotated image to {output_path}")

        return {
            "success": True,
            "total_detections": len(detections),
            "detections": detections,
            "class_counts": class_counts,
            "processing_time": round(elapsed_time, 4),
            "processing_time_ms": round(elapsed_time * 1000, 1),
            "image_dimensions": {"width": width, "height": height},
            "conf_threshold": conf_threshold,
        }
