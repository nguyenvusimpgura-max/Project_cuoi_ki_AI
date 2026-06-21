"""
Canteen Project - VSCode/Local inference
Chạy CNN + OpenCV crop khay + Roboflow đếm trứng trên toàn ảnh gốc để xuất hóa đơn.

Ví dụ:
    python main.py --image data/test_trays/test002.jpg
    python main.py --input-dir data/test_trays
    python main.py --input-dir data/test_trays --no-egg
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv là optional
    load_dotenv = None


# =========================
# CẤU HÌNH CỐ ĐỊNH
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent

TRAY_SIZE = (800, 600)       # (width, height) ảnh khay trước khi crop
CNN_IMG_SIZE = (224, 224)    # (width, height) crop đưa vào CNN
CONF_THRESHOLD = 0.45
EGG_CONF_THRESHOLD = 0.35
DEFAULT_API_URL = "https://serverless.roboflow.com"
DEFAULT_MODEL_ID = "egg2-3omrr/1"

COMPARTMENTS = {
    "o_lon_tren_trai": [45, 280, 55, 385],
    "o_lon_tren_phai": [45, 280, 480, 750],
    "o_nho_duoi_trai": [335, 550, 45, 270],
    "o_nho_duoi_giua": [335, 550, 315, 495],
    "o_nho_duoi_phai": [335, 550, 540, 760],
}

SLOT_LABELS_VI = {
    "o_lon_tren_trai": "Ô lớn trên trái",
    "o_lon_tren_phai": "Ô lớn trên phải",
    "o_nho_duoi_trai": "Ô nhỏ dưới trái",
    "o_nho_duoi_giua": "Ô nhỏ dưới giữa",
    "o_nho_duoi_phai": "Ô nhỏ dưới phải",
}

DEFAULT_MENU = {
    "com": 10000,
    "com_trang": 10000,
    "dau_hu_sot_ca": 25000,
    "ca_hu_kho": 30000,
    "thit_kho_trung": 25000,  # gộp về thit_kho, trứng tính riêng
    "thit_kho": 25000,
    "canh_chua_co_ca": 25000,
    "canh_chua_khong_ca": 10000,
    "suon_nuong": 30000,
    "canh_rau": 7000,
    "rau_xao": 10000,
    "trung_chien": 25000,
    "trung_chien_thit": 25000,
    "khay_trong": 0,
    "khay_rong": 0,
    "empty": 0,
    "background": 0,
}

# Nếu tên class CNN khác tên trong menu thì map ở đây.
CLASS_ALIASES = {
    "cơm_trắng": "com",
    "com_trang": "com",
    "com": "com",
    "dau_hu_sot_ca": "dau_hu_sot_ca",
    "dau_hu": "dau_hu_sot_ca",
    "ca_hu_kho": "ca_hu_kho",
    "thit_kho_trung": "thit_kho",
    "thit_kho": "thit_kho",
    "canh_chua_co_ca": "canh_chua_co_ca",
    "canh_chua_khong_ca": "canh_chua_khong_ca",
    "suon_nuong": "suon_nuong",
    "canh_rau": "canh_rau",
    "rau_xao": "rau_xao",
    "trung_chien": "trung_chien",
    "trung_chien_thit": "trung_chien_thit",
    "khay_trong": "khay_trong",
    "khay_rong": "khay_trong",
    "empty": "khay_trong",
    "background": "khay_trong",
}

EGG_CANDIDATE_CLASSES = {"thit_kho", "thit_kho_trung"}  # giữ để tương thích, không còn dùng để quyết định gọi Roboflow
THIT_KHO_TRUNG_BASE_PRICE = 25000  # giữ để tương thích, thit_kho_trung đã gộp về thit_kho
EXTRA_EGG_PRICE = 6000

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# =========================
# HÀM PHỤ TRỢ
# =========================

def ensure_dirs(root: Path) -> Dict[str, Path]:
    paths = {
        "model_dir": root / "models" / "cnn",
        "raw_trays_dir": root / "data" / "raw_trays",
        "test_trays_dir": root / "data" / "test_trays",
        "empty_trays_dir": root / "data" / "empty_trays",
        "inference_dir": root / "inference",
        "cropped_dir": root / "inference" / "cropped_slots",
        "cropped_224_dir": root / "inference" / "cropped_slots_224",
        "debug_boxes_dir": root / "inference" / "debug_boxes",
        "egg_debug_dir": root / "inference" / "egg_debug",
        "egg_input_dir": root / "inference" / "egg_debug" / "input_images",
        "egg_annotated_dir": root / "inference" / "egg_debug" / "annotated",
        "egg_json_dir": root / "inference" / "egg_debug" / "result_json",
        "predictions_dir": root / "inference" / "predictions",
        "bills_dir": root / "outputs" / "bills",
        "logs_dir": root / "outputs" / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def list_files(folder: Path, exts: Iterable[str] = IMAGE_EXTS) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    files: List[Path] = []
    for ext in exts:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def format_vnd(n: int | float) -> str:
    return f"{int(n):,}".replace(",", ".") + "đ"


def normalize_label(label: Any) -> str:
    s = str(label).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def canonical_class(pred_class: Any) -> str:
    pred_class = normalize_label(pred_class)
    return CLASS_ALIASES.get(pred_class, pred_class)


def read_image_cv2(image_path: Path) -> np.ndarray:
    """Đọc ảnh bằng cách hỗ trợ tốt hơn đường dẫn Unicode trên Windows."""
    image_path = Path(image_path)
    data = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")
    return img


def write_image_cv2(out_path: Path, img_bgr: np.ndarray) -> None:
    """Ghi ảnh hỗ trợ tốt hơn đường dẫn Unicode trên Windows."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        raise ValueError(f"Không encode được ảnh: {out_path}")
    buf.tofile(str(out_path))


def safe_json_dump(data: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_egg_class(cls_name: Any) -> str:
    s = str(cls_name).strip().lower().replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    if s in ["egg whole", "whole egg", "eggwhole"]:
        return "egg whole"
    if s in ["egg half", "half egg", "egghalf"]:
        return "egg half"
    return s


def has_rescaling_layer(model: Any) -> bool:
    for layer in getattr(model, "layers", []):
        if layer.__class__.__name__.lower() == "rescaling":
            return True
        if "rescaling" in getattr(layer, "name", "").lower():
            return True
        if hasattr(layer, "layers") and has_rescaling_layer(layer):
            return True
    return False


# =========================
# INFERENCE CLASS
# =========================

class CanteenInferencer:
    def __init__(
        self,
        root: Path = PROJECT_ROOT,
        model_path: Optional[Path] = None,
        class_names_path: Optional[Path] = None,
        menu_path: Optional[Path] = None,
        use_egg: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.paths = ensure_dirs(self.root)

        if load_dotenv is not None:
            load_dotenv(self.root / ".env")

        self.model_path = Path(model_path) if model_path else self.paths["model_dir"] / "final_canteen_cnn.keras"
        self.class_names_path = Path(class_names_path) if class_names_path else self.paths["model_dir"] / "class_names.json"
        self.menu_path = Path(menu_path) if menu_path else self.paths["model_dir"] / "menu.json"

        self.model = self._load_cnn_model()
        self.class_names = self._load_class_names()
        self.menu = self._load_menu()
        self.model_has_rescaling = has_rescaling_layer(self.model)

        self.rf_client = None
        self.roboflow_model_id = (os.getenv("ROBOFLOW_MODEL_ID") or os.getenv("DEFAULT_MODEL_ID") or DEFAULT_MODEL_ID).strip()
        self.roboflow_api_url = (os.getenv("ROBOFLOW_API_URL") or os.getenv("DEFAULT_API_URL") or DEFAULT_API_URL).strip()
        self.roboflow_api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()

        if use_egg:
            self._init_roboflow_if_possible()

        print("ROOT:", self.root)
        print("CNN model:", self.model_path)
        print("Số class:", len(self.class_names))
        print("Model có Rescaling bên trong?:", self.model_has_rescaling)
        print("Roboflow egg detection:", "ON" if self.rf_client else "OFF")

    def _load_cnn_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Thiếu model CNN: {self.model_path}")
        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow import keras
        except ImportError as exc:
            raise ImportError(
                "Chưa cài TensorFlow. Hãy chạy: pip install -r requirements.txt"
            ) from exc
        return keras.models.load_model(self.model_path)

    def _load_class_names(self) -> List[str]:
        if not self.class_names_path.exists():
            raise FileNotFoundError(f"Thiếu class_names.json: {self.class_names_path}")
        with open(self.class_names_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
        if isinstance(class_names, dict):
            class_names = [class_names[str(i)] for i in range(len(class_names))]
        return list(class_names)

    def _load_menu(self) -> Dict[str, int]:
        if self.menu_path.exists():
            with open(self.menu_path, "r", encoding="utf-8") as f:
                menu = json.load(f)
        else:
            menu = DEFAULT_MENU.copy()
            safe_json_dump(menu, self.menu_path)
        # Bổ sung key thiếu để tránh giá bị 0 do lệch tên class.
        merged = DEFAULT_MENU.copy()
        merged.update({canonical_class(k): int(v) for k, v in menu.items()})
        return merged

    def _init_roboflow_if_possible(self) -> None:
        if not self.roboflow_api_key:
            print("Không thấy ROBOFLOW_API_KEY trong .env nên sẽ bỏ qua bước đếm trứng.")
            return
        try:
            from inference_sdk import InferenceHTTPClient
        except ImportError:
            print("Chưa cài inference-sdk nên sẽ bỏ qua bước đếm trứng.")
            return
        self.rf_client = InferenceHTTPClient(
            api_url=self.roboflow_api_url,
            api_key=self.roboflow_api_key,
        )

    def crop_slots_from_tray(self, image_path: Path, save: bool = True, pad: int = 0):
        image_path = Path(image_path)
        img = read_image_cv2(image_path)
        target_w, target_h = TRAY_SIZE
        resized_img = cv2.resize(img, (target_w, target_h))
        debug_img = resized_img.copy()

        crops: Dict[str, np.ndarray] = {}
        crops_224: Dict[str, np.ndarray] = {}

        raw_out_dir = self.paths["cropped_dir"] / image_path.stem
        resized_out_dir = self.paths["cropped_224_dir"] / image_path.stem

        if save:
            raw_out_dir.mkdir(parents=True, exist_ok=True)
            resized_out_dir.mkdir(parents=True, exist_ok=True)

        for slot_name, box in COMPARTMENTS.items():
            ymin, ymax, xmin, xmax = box
            ymin2 = max(0, ymin + pad)
            ymax2 = min(target_h, ymax - pad)
            xmin2 = max(0, xmin + pad)
            xmax2 = min(target_w, xmax - pad)

            crop = resized_img[ymin2:ymax2, xmin2:xmax2]
            crop_224 = cv2.resize(crop, CNN_IMG_SIZE)

            crops[slot_name] = crop
            crops_224[slot_name] = crop_224

            if save:
                write_image_cv2(raw_out_dir / f"{slot_name}.jpg", crop)
                write_image_cv2(resized_out_dir / f"{slot_name}_224.jpg", crop_224)

            cv2.rectangle(debug_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                debug_img,
                slot_name,
                (xmin, max(20, ymin - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

        if save:
            write_image_cv2(self.paths["debug_boxes_dir"] / f"{image_path.stem}_debug.jpg", debug_img)

        return crops, crops_224, debug_img, resized_img

    def prepare_crop_for_cnn(self, crop_bgr: np.ndarray) -> np.ndarray:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_rgb = cv2.resize(crop_rgb, CNN_IMG_SIZE)
        arr = crop_rgb.astype("float32")
        if not self.model_has_rescaling:
            arr = arr / 255.0
        return np.expand_dims(arr, axis=0)

    def predict_crop_cnn(self, crop_bgr: np.ndarray, top_k: int = 3) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        x = self.prepare_crop_for_cnn(crop_bgr)
        probs = self.model.predict(x, verbose=0)[0]
        top_idx = probs.argsort()[-top_k:][::-1]

        top_preds: List[Dict[str, Any]] = []
        for idx in top_idx:
            raw_label = self.class_names[int(idx)]
            top_preds.append(
                {
                    "class": normalize_label(raw_label),
                    "raw_class": raw_label,
                    "confidence": float(probs[int(idx)]),
                }
            )
        return top_preds[0], top_preds

    def is_candidate_for_egg_detection(self, pred_class: str, top_preds: Optional[List[Dict[str, Any]]] = None) -> bool:
        pred_class = canonical_class(pred_class)
        if pred_class in EGG_CANDIDATE_CLASSES:
            return True
        if top_preds:
            for p in top_preds:
                c = canonical_class(p.get("class", ""))
                if c in EGG_CANDIDATE_CLASSES:
                    return True
        return False

    def draw_egg_predictions(self, crop_bgr: np.ndarray, predictions: List[Dict[str, Any]], out_path: Path) -> np.ndarray:
        img = crop_bgr.copy()
        for p in predictions:
            conf = float(p.get("confidence", 0))
            if conf < EGG_CONF_THRESHOLD:
                continue

            cls = parse_egg_class(p.get("class", ""))
            x = float(p.get("x", 0))
            y = float(p.get("y", 0))
            w = float(p.get("width", 0))
            h = float(p.get("height", 0))

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                img,
                f"{cls} {conf:.2f}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        write_image_cv2(out_path, img)
        return img

    def count_eggs_roboflow(self, image_bgr: np.ndarray, image_name: str = "tray.jpg", save_debug: bool = True) -> Dict[str, Any]:
        """Đếm trứng bằng Roboflow trên TOÀN BỘ ảnh gốc.

        Hàm này nhận ảnh trước khi resize/crop. Vì vậy thứ tự đúng là:
        Roboflow đếm trứng toàn ảnh gốc -> sau đó mới resize/crop cho CNN.
        """
        input_h, input_w = image_bgr.shape[:2]

        if self.rf_client is None:
            return {
                "egg_whole": 0,
                "egg_half": 0,
                "egg_equiv": 0.0,
                "billable_eggs": 0,
                "egg_price": 0,
                "egg_price_vnd": format_vnd(0),
                "predictions": [],
                "confidence_threshold": EGG_CONF_THRESHOLD,
                "input_width": int(input_w),
                "input_height": int(input_h),
                "skipped": True,
                "reason": "Roboflow is OFF. Add ROBOFLOW_API_KEY to .env or remove --no-egg.",
            }

        image_path = self.paths["egg_input_dir"] / image_name
        write_image_cv2(image_path, image_bgr)

        result = self.rf_client.infer(str(image_path), model_id=self.roboflow_model_id)
        predictions = result.get("predictions", []) if isinstance(result, dict) else []

        egg_whole = 0
        egg_half = 0
        kept_predictions: List[Dict[str, Any]] = []

        for p in predictions:
            conf = float(p.get("confidence", 0))
            if conf < EGG_CONF_THRESHOLD:
                continue

            cls = parse_egg_class(p.get("class", ""))
            if cls == "egg whole":
                egg_whole += 1
            elif cls == "egg half":
                egg_half += 1
            else:
                continue

            p2 = dict(p)
            p2["class_normalized"] = cls
            kept_predictions.append(p2)

        egg_equiv = egg_whole + egg_half * 0.5
        billable_eggs = int(math.ceil(egg_equiv)) if egg_equiv > 0 else 0
        egg_price = int(billable_eggs * EXTRA_EGG_PRICE)

        egg_info = {
            "scope": "whole_original_image_before_resize",
            "egg_whole": int(egg_whole),
            "egg_half": int(egg_half),
            "egg_equiv": float(egg_equiv),
            "billable_eggs": int(billable_eggs),
            "egg_price": int(egg_price),
            "egg_price_vnd": format_vnd(egg_price),
            "unit_price": int(EXTRA_EGG_PRICE),
            "unit_price_vnd": format_vnd(EXTRA_EGG_PRICE),
            "predictions": kept_predictions,
            "confidence_threshold": EGG_CONF_THRESHOLD,
            "input_width": int(input_w),
            "input_height": int(input_h),
            "skipped": False,
        }

        if save_debug:
            stem = Path(image_name).stem
            safe_json_dump(egg_info, self.paths["egg_json_dir"] / f"{stem}_whole_image_egg.json")
            self.draw_egg_predictions(
                image_bgr,
                kept_predictions,
                self.paths["egg_annotated_dir"] / f"{stem}_whole_image_egg_annotated.jpg",
            )

        return egg_info

    def price_for_prediction(self, pred_class: str, egg_info: Optional[Dict[str, Any]] = None) -> Tuple[str, int, str]:
        """Tính giá món theo CNN.

        Lưu ý: thit_kho_trung đã được canonical_class() gộp thành thit_kho.
        Trứng không còn gắn vào ô thịt kho nữa; trứng được cộng thành dòng riêng của toàn ảnh.
        """
        final_class = canonical_class(pred_class)
        price = int(self.menu.get(final_class, DEFAULT_MENU.get(final_class, 0)))

        note = ""
        if normalize_label(pred_class) == "thit_kho_trung":
            note = "Đã gộp thit_kho_trung về thit_kho; trứng được tính riêng bằng Roboflow toàn ảnh."

        return final_class, price, note

    def price_for_eggs(self, egg_info: Optional[Dict[str, Any]]) -> Tuple[int, str]:
        egg_info = egg_info or {}
        billable_eggs = int(egg_info.get("billable_eggs", 0) or 0)
        price = int(billable_eggs * EXTRA_EGG_PRICE)
        note = (
            f"Roboflow đếm trên toàn bộ ảnh gốc trước khi resize: "
            f"whole={int(egg_info.get('egg_whole', 0) or 0)}, "
            f"half={int(egg_info.get('egg_half', 0) or 0)}, "
            f"quy đổi={float(egg_info.get('egg_equiv', 0.0) or 0.0):g}, "
            f"tính tiền={billable_eggs} trứng."
        )
        return price, note

    def infer_one_tray(self, image_path: Path, save: bool = True, show: bool = False) -> Dict[str, Any]:
        image_path = Path(image_path)

        # Yêu cầu mới: Roboflow đếm trứng trên toàn bộ ảnh gốc TRƯỚC KHI resize/crop.
        original_img = read_image_cv2(image_path)
        egg_info = self.count_eggs_roboflow(
            original_img,
            image_name=f"{image_path.stem}_whole_original.jpg",
            save_debug=save,
        )
        egg_price, egg_note = self.price_for_eggs(egg_info)

        # Sau khi đã đếm trứng xong mới resize/crop để CNN phân loại từng ô.
        crops, _crops_224, debug_img, _resized_img = self.crop_slots_from_tray(image_path, save=save, pad=0)

        items: List[Dict[str, Any]] = []
        dish_total = 0

        for slot_name, crop in crops.items():
            pred, top_preds = self.predict_crop_cnn(crop, top_k=3)
            pred_class = canonical_class(pred["class"])

            final_class, price, note = self.price_for_prediction(pred_class)
            dish_total += price

            items.append(
                {
                    "slot": slot_name,
                    "slot_vi": SLOT_LABELS_VI.get(slot_name, slot_name),
                    "pred_class": pred_class,
                    "raw_pred_class": pred["raw_class"],
                    "confidence": float(pred["confidence"]),
                    "top3": top_preds,
                    "final_class": final_class,
                    "price": int(price),
                    "price_vnd": format_vnd(price),
                    "note": note,
                }
            )

        total = int(dish_total + egg_price)
        bill = {
            "image_name": image_path.name,
            "image_path": str(image_path),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items": items,
            "egg_info": egg_info,
            "egg_item": {
                "final_class": "trung_roboflow",
                "quantity_equiv": float(egg_info.get("egg_equiv", 0.0) or 0.0),
                "quantity_billable": int(egg_info.get("billable_eggs", 0) or 0),
                "unit_price": int(EXTRA_EGG_PRICE),
                "unit_price_vnd": format_vnd(EXTRA_EGG_PRICE),
                "price": int(egg_price),
                "price_vnd": format_vnd(egg_price),
                "note": egg_note,
            },
            "dish_total": int(dish_total),
            "dish_total_vnd": format_vnd(dish_total),
            "egg_total": int(egg_price),
            "egg_total_vnd": format_vnd(egg_price),
            "total": total,
            "total_vnd": format_vnd(total),
        }

        if save:
            bill_json = self.paths["bills_dir"] / f"{image_path.stem}_bill.json"
            safe_json_dump(bill, bill_json)

            bill_txt = self.paths["bills_dir"] / f"{image_path.stem}_bill.txt"
            with open(bill_txt, "w", encoding="utf-8") as f:
                f.write(f"HÓA ĐƠN: {image_path.name}\n")
                f.write("=" * 50 + "\n")
                for item in items:
                    f.write(
                        f"{item['slot_vi']}: {item['final_class']} - {item['price_vnd']} - conf={item['confidence']:.3f}\n"
                    )
                    if item["note"]:
                        f.write(f"  Ghi chú: {item['note']}\n")

                f.write("-" * 50 + "\n")
                f.write(
                    f"Trứng Roboflow toàn ảnh: "
                    f"whole={egg_info.get('egg_whole', 0)}, "
                    f"half={egg_info.get('egg_half', 0)}, "
                    f"quy đổi={float(egg_info.get('egg_equiv', 0.0) or 0.0):g}, "
                    f"tính tiền={int(egg_info.get('billable_eggs', 0) or 0)} x {format_vnd(EXTRA_EGG_PRICE)} "
                    f"= {format_vnd(egg_price)}\n"
                )
                f.write("=" * 50 + "\n")
                f.write(f"TỔNG MÓN: {bill['dish_total_vnd']}\n")
                f.write(f"TỔNG TRỨNG: {bill['egg_total_vnd']}\n")
                f.write(f"TỔNG: {bill['total_vnd']}\n")

        self.print_bill(bill)

        if show:
            try:
                import matplotlib.pyplot as plt

                img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(10, 7))
                plt.imshow(img_rgb)
                plt.axis("off")
                plt.title(f"Debug boxes: {image_path.name}")
                plt.show()
            except ImportError:
                print("Chưa cài matplotlib nên không hiển thị ảnh debug được.")

        return bill

    @staticmethod
    def print_bill(bill: Dict[str, Any]) -> None:
        print("\n" + "=" * 70)
        print("Ảnh:", bill["image_name"])
        print("=" * 70)
        for item in bill["items"]:
            print(
                f"{item['slot_vi']:<18} | {item['final_class']:<22} | "
                f"{item['price_vnd']:<10} | conf={item['confidence']:.3f}"
            )
            if item["note"]:
                print("  ↳", item["note"])

        egg_info = bill.get("egg_info", {})
        egg_item = bill.get("egg_item", {})
        print("-" * 70)
        print(
            f"{'Trứng Roboflow':<18} | {'trung_roboflow':<22} | "
            f"{egg_item.get('price_vnd', format_vnd(0)):<10} | "
            f"whole={egg_info.get('egg_whole', 0)}, "
            f"half={egg_info.get('egg_half', 0)}, "
            f"quy đổi={float(egg_info.get('egg_equiv', 0.0) or 0.0):g}"
        )
        print("-" * 70)
        print("TỔNG MÓN:", bill.get("dish_total_vnd", format_vnd(0)))
        print("TỔNG TRỨNG:", bill.get("egg_total_vnd", format_vnd(0)))
        print("TỔNG:", bill["total_vnd"])

    def infer_batch(self, input_dir: Path, save: bool = True, show: bool = False) -> pd.DataFrame:
        images = list_files(Path(input_dir))
        if not images:
            raise FileNotFoundError(f"Không có ảnh trong thư mục: {input_dir}")

        all_bills: List[Dict[str, Any]] = []
        for image_path in images:
            print("\nĐang xử lý:", image_path.name)
            bill = self.infer_one_tray(image_path, save=save, show=show)
            all_bills.append(bill)

        summary_rows: List[Dict[str, Any]] = []
        for bill in all_bills:
            row: Dict[str, Any] = {
                "image_name": bill["image_name"],
                "dish_total": bill.get("dish_total", 0),
                "egg_equiv": bill.get("egg_info", {}).get("egg_equiv", 0),
                "billable_eggs": bill.get("egg_info", {}).get("billable_eggs", 0),
                "egg_total": bill.get("egg_total", 0),
                "total": bill["total"],
                "total_vnd": bill["total_vnd"],
            }
            for item in bill["items"]:
                row[item["slot"]] = item["final_class"]
                row[item["slot"] + "_price"] = item["price"]
                row[item["slot"] + "_conf"] = item["confidence"]
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        if save:
            summary_csv = self.paths["logs_dir"] / "inference_summary.csv"
            summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
            print("\nĐã lưu summary:", summary_csv)
        return summary_df


# =========================
# CLI
# =========================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nhận diện món ăn trên khay và tính tiền.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Thư mục project. Mặc định là thư mục chứa main.py.")
    parser.add_argument("--image", type=Path, help="Đường dẫn 1 ảnh khay cần xử lý.")
    parser.add_argument("--input-dir", type=Path, help="Thư mục chứa nhiều ảnh khay cần xử lý.")
    parser.add_argument("--model", type=Path, help="Đường dẫn file .keras nếu không dùng mặc định.")
    parser.add_argument("--class-names", type=Path, help="Đường dẫn class_names.json nếu không dùng mặc định.")
    parser.add_argument("--menu", type=Path, help="Đường dẫn menu.json nếu không dùng mặc định.")
    parser.add_argument("--no-egg", action="store_true", help="Tắt Roboflow đếm trứng, chỉ dùng CNN.")
    parser.add_argument("--show", action="store_true", help="Hiển thị ảnh debug bằng matplotlib.")
    parser.add_argument("--no-save", action="store_true", help="Không lưu crop/debug/bill ra file.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    inferencer = CanteenInferencer(
        root=args.root,
        model_path=args.model,
        class_names_path=args.class_names,
        menu_path=args.menu,
        use_egg=not args.no_egg,
    )

    save = not args.no_save

    if args.image:
        inferencer.infer_one_tray(args.image, save=save, show=args.show)
        return

    if args.input_dir:
        df = inferencer.infer_batch(args.input_dir, save=save, show=args.show)
        print("\nBảng tổng hợp:")
        print(df.to_string(index=False))
        return

    # Nếu không truyền gì, chạy thử thư mục data/test_trays trước, nếu rỗng thì raw_trays.
    default_input = inferencer.paths["test_trays_dir"]
    if not list_files(default_input):
        default_input = inferencer.paths["raw_trays_dir"]

    print(f"Không truyền --image/--input-dir nên chạy mặc định: {default_input}")
    df = inferencer.infer_batch(default_input, save=save, show=args.show)
    print("\nBảng tổng hợp:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
