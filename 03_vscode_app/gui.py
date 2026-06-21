"""
GUI desktop cho Canteen Project.

Chạy:
    python gui.py

Tính năng chính:
- Chọn ảnh khay, xoay, zoom, kéo-thả để căn ảnh vào khung CV cố định.
- Bắt buộc bấm "Áp dụng khung CV" trước khi chạy model.
- Roboflow đếm trứng trên toàn ảnh đã căn khung trước khi CNN crop/resize từng ô.
- Đọc cấu hình Roboflow từ .env.
- Hiển thị ảnh kết quả, bảng từng ô, giá và tổng tiền.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import customtkinter as ctk
import cv2
from PIL import Image, ImageOps, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from main import (
    CanteenInferencer,
    COMPARTMENTS,
    DEFAULT_API_URL,
    DEFAULT_MODEL_ID,
    PROJECT_ROOT,
    TRAY_SIZE,
    format_vnd,
    read_image_cv2,
    write_image_cv2,
)


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


IMAGE_FILETYPES = [
    ("Ảnh", "*.jpg *.jpeg *.png *.webp *.bmp"),
    ("Tất cả file", "*.*"),
]


class CanteenGUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Canteen Project - Nhận diện món ăn & tính tiền")
        self.geometry("1420x900")
        self.minsize(1180, 780)

        self.project_root = PROJECT_ROOT
        self.env_path = self.project_root / ".env"

        self.selected_image: Optional[Path] = None
        self.preview_source_path: Optional[Path] = None
        self.aligned_image_path: Optional[Path] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.inferencer: Optional[CanteenInferencer] = None
        self.worker: Optional[threading.Thread] = None
        self.result_queue: queue.Queue = queue.Queue()

        self.image_rotation_deg = 0
        self.zoom_percent = 100
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_start: Optional[Tuple[int, int]] = None
        self.preview_mode = "empty"  # empty | align | result
        self.cv_frame_applied = False
        self.last_align_geometry: Optional[Dict[str, float]] = None

        self.env_config = self._read_roboflow_env()

        self.image_path_var = tk.StringVar(value="Chưa chọn ảnh")
        self.status_var = tk.StringVar(value="Sẵn sàng. Hãy chọn ảnh khay để nhận diện.")
        self.total_var = tk.StringVar(value="TỔNG: 0đ")
        self.zoom_var = tk.StringVar(value="Zoom: 100%")
        self.rotation_var = tk.StringVar(value="Góc xoay: 0°")
        self.cv_state_var = tk.StringVar(value="Khung CV: chưa có ảnh")
        self.rf_status_var = tk.StringVar(value=self._make_rf_status_text())
        self.use_egg_var = tk.BooleanVar(value=bool(self.env_config["api_key"]))

        self._build_layout()
        self._set_run_buttons_state("disabled")
        self.after(200, self._poll_result_queue)

    # =========================
    # ENV / CONFIG
    # =========================

    def _parse_env_file(self, path: Path) -> Dict[str, str]:
        values: Dict[str, str] = {}
        if not path.exists():
            return values

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
        return values

    def _read_roboflow_env(self) -> Dict[str, str]:
        file_values = self._parse_env_file(self.env_path)
        api_key = os.getenv("ROBOFLOW_API_KEY") or file_values.get("ROBOFLOW_API_KEY", "")
        model_id = (
            os.getenv("ROBOFLOW_MODEL_ID")
            or file_values.get("ROBOFLOW_MODEL_ID")
            or os.getenv("DEFAULT_MODEL_ID")
            or file_values.get("DEFAULT_MODEL_ID")
            or DEFAULT_MODEL_ID
        )
        api_url = (
            os.getenv("ROBOFLOW_API_URL")
            or file_values.get("ROBOFLOW_API_URL")
            or os.getenv("DEFAULT_API_URL")
            or file_values.get("DEFAULT_API_URL")
            or DEFAULT_API_URL
        )
        return {
            "api_key": api_key.strip(),
            "model_id": model_id.strip(),
            "api_url": api_url.strip(),
        }

    @staticmethod
    def _mask_key(key: str) -> str:
        if not key:
            return "chưa có"
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:4]}...{key[-4:]}"

    def _make_rf_status_text(self) -> str:
        api_key = self.env_config["api_key"]
        return (
            f"Roboflow: {'sẵn sàng' if api_key else 'chưa có API key'}\n"
            f"API key: {self._mask_key(api_key)}\n"
            f"Model: {self.env_config['model_id']}\n"
            f"URL: {self.env_config['api_url']}"
        )

    def reload_env(self) -> None:
        self.env_config = self._read_roboflow_env()
        self.rf_status_var.set(self._make_rf_status_text())
        self.use_egg_var.set(bool(self.env_config["api_key"]))
        self.inferencer = None
        self.status_var.set("Đã reload .env. Lần nhận diện tiếp theo sẽ load lại model.")

    # =========================
    # UI
    # =========================

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, corner_radius=18)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(16, 8))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Canteen Project",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(14, 0))
        ctk.CTkLabel(
            header,
            text="Chọn ảnh → kéo/zoom căn vào khung CV → áp dụng khung → chạy model.",
            text_color=("#667085", "#B0B7C3"),
        ).grid(row=1, column=0, sticky="w", padx=18, pady=(0, 14))

        self.run_button = ctk.CTkButton(header, text="Nhận diện", width=140, command=self.run_inference)
        self.run_button.grid(row=0, column=1, rowspan=2, sticky="e", padx=18, pady=14)

        left = ctk.CTkFrame(self, corner_radius=18)
        left.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(0, 16))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(4, weight=1)

        right = ctk.CTkFrame(self, corner_radius=18)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=(0, 16))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(3, weight=1)

        file_bar = ctk.CTkFrame(left, fg_color="transparent")
        file_bar.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        file_bar.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(file_bar, text="Chọn ảnh", width=115, command=self.choose_image).grid(row=0, column=0, padx=(0, 10))
        ctk.CTkLabel(file_bar, textvariable=self.image_path_var, anchor="w").grid(row=0, column=1, sticky="ew")

        rotate_bar = ctk.CTkFrame(left, fg_color="transparent")
        rotate_bar.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        rotate_bar.grid_columnconfigure((0, 1, 2), weight=1)
        ctk.CTkButton(rotate_bar, text="⟲ Xoay trái", command=self.rotate_left).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(rotate_bar, text="Xoay phải ⟳", command=self.rotate_right).grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkButton(rotate_bar, text="Reset xoay", command=self.reset_rotation).grid(row=0, column=2, sticky="ew", padx=(0, 8))
        ctk.CTkLabel(rotate_bar, textvariable=self.rotation_var, width=110).grid(row=0, column=3, sticky="e")

        zoom_bar = ctk.CTkFrame(left, fg_color="transparent")
        zoom_bar.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))
        zoom_bar.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(zoom_bar, text="-", width=38, command=self.zoom_out).grid(row=0, column=0, padx=(0, 8))
        self.zoom_slider = ctk.CTkSlider(
            zoom_bar,
            from_=60,
            to=260,
            number_of_steps=40,
            command=self._on_zoom_slider_change,
        )
        self.zoom_slider.set(100)
        self.zoom_slider.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkButton(zoom_bar, text="+", width=38, command=self.zoom_in).grid(row=0, column=2, padx=(0, 8))
        ctk.CTkButton(zoom_bar, text="100%", width=62, command=self.reset_zoom).grid(row=0, column=3, padx=(0, 8))
        ctk.CTkLabel(zoom_bar, textvariable=self.zoom_var, width=95).grid(row=0, column=4, sticky="e")

        cv_bar = ctk.CTkFrame(left, fg_color="transparent")
        cv_bar.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 10))
        cv_bar.grid_columnconfigure((0, 1), weight=1)
        self.apply_cv_button = ctk.CTkButton(cv_bar, text="Áp dụng khung CV", command=self.apply_cv_frame)
        self.apply_cv_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(cv_bar, text="Reset căn khung", command=self.reset_alignment).grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkLabel(cv_bar, textvariable=self.cv_state_var, anchor="e").grid(row=0, column=2, sticky="e")

        preview_frame = ctk.CTkFrame(left, corner_radius=14, fg_color=("#F7F8FA", "#151922"))
        preview_frame.grid(row=4, column=0, sticky="nsew", padx=14, pady=(0, 14))
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_canvas = tk.Canvas(preview_frame, highlightthickness=0, bg="#111827")
        self.preview_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.preview_canvas.bind("<Configure>", lambda _event: self._redraw_current_preview())
        self.preview_canvas.bind("<ButtonPress-1>", self._start_pan)
        self.preview_canvas.bind("<B1-Motion>", self._drag_pan)
        self.preview_canvas.bind("<ButtonRelease-1>", self._stop_pan)
        self.preview_canvas.bind("<MouseWheel>", self._mousewheel_zoom)
        self.preview_canvas.bind("<Button-4>", self._mousewheel_zoom)
        self.preview_canvas.bind("<Button-5>", self._mousewheel_zoom)

        config_frame = ctk.CTkFrame(right, corner_radius=16)
        config_frame.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))
        config_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(config_frame, text="Roboflow / trứng toàn ảnh", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 6)
        )
        ctk.CTkCheckBox(
            config_frame,
            text="Bật Roboflow đếm trứng toàn ảnh đã căn khung",
            variable=self.use_egg_var,
            command=self._reset_inferencer_when_option_changed,
        ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 8))
        ctk.CTkLabel(
            config_frame,
            textvariable=self.rf_status_var,
            justify="left",
            wraplength=470,
            text_color=("#475467", "#B0B7C3"),
        ).grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))

        env_buttons = ctk.CTkFrame(config_frame, fg_color="transparent")
        env_buttons.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 14))
        env_buttons.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(env_buttons, text="Reload .env", command=self.reload_env).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(env_buttons, text="Mở .env", command=lambda: self.open_path(self.env_path)).grid(row=0, column=1, sticky="ew")

        action_frame = ctk.CTkFrame(right, corner_radius=16)
        action_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))
        action_frame.grid_columnconfigure(0, weight=1)
        self.run_button_action = ctk.CTkButton(action_frame, text="Nhận diện ảnh đã căn khung", command=self.run_inference)
        self.run_button_action.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        ctk.CTkButton(action_frame, text="Mở thư mục hóa đơn", command=self.open_bills_folder).grid(
            row=1, column=0, sticky="ew", padx=14, pady=(0, 8)
        )
        ctk.CTkButton(action_frame, text="Mở file chỉnh giá menu", command=self.open_menu_file).grid(
            row=2, column=0, sticky="ew", padx=14, pady=(0, 14)
        )

        self.status_label = ctk.CTkLabel(right, textvariable=self.status_var, justify="left", wraplength=470)
        self.status_label.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))

        result_frame = ctk.CTkFrame(right, corner_radius=16)
        result_frame.grid(row=3, column=0, sticky="nsew", padx=14, pady=(0, 10))
        result_frame.grid_columnconfigure(0, weight=1)
        result_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(result_frame, text="Kết quả món + trứng riêng", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(14, 8)
        )

        table_holder = ctk.CTkFrame(result_frame, corner_radius=10)
        table_holder.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        table_holder.grid_columnconfigure(0, weight=1)
        table_holder.grid_rowconfigure(0, weight=1)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Treeview", rowheight=30, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

        columns = ("slot", "pred", "dish", "conf", "eggs", "price")
        self.result_table = ttk.Treeview(table_holder, columns=columns, show="headings", height=10)
        headings = {
            "slot": "Ô",
            "pred": "CNN",
            "dish": "Món tính tiền",
            "conf": "Conf",
            "eggs": "Trứng",
            "price": "Giá",
        }
        widths = {
            "slot": 125,
            "pred": 110,
            "dish": 135,
            "conf": 62,
            "eggs": 62,
            "price": 90,
        }
        for col in columns:
            self.result_table.heading(col, text=headings[col])
            anchor = "e" if col == "price" else "center" if col in {"conf", "eggs"} else "w"
            self.result_table.column(col, width=widths[col], anchor=anchor)
        self.result_table.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(table_holder, orient="vertical", command=self.result_table.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        self.result_table.configure(yscrollcommand=y_scroll.set)

        self.total_label = ctk.CTkLabel(
            right,
            textvariable=self.total_var,
            font=ctk.CTkFont(size=23, weight="bold"),
        )
        self.total_label.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 14))

        self._draw_empty_preview()

    # =========================
    # Basic actions
    # =========================

    def choose_image(self) -> None:
        image_path = filedialog.askopenfilename(
            title="Chọn ảnh khay",
            initialdir=str(self.project_root / "data" / "test_trays"),
            filetypes=IMAGE_FILETYPES,
        )
        if not image_path:
            return

        self.selected_image = Path(image_path)
        self.preview_source_path = None
        self.aligned_image_path = None
        self.preview_mode = "align"
        self.image_rotation_deg = 0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._set_zoom(100, mark_dirty=False)
        self.cv_frame_applied = False
        self.image_path_var.set(str(self.selected_image))
        self.rotation_var.set("Góc xoay: 0°")
        self.cv_state_var.set("Khung CV: chưa áp dụng")
        self.status_var.set("Đã chọn ảnh. Kéo ảnh để căn vào khung xanh, zoom nếu cần, rồi bấm 'Áp dụng khung CV'.")
        self._set_run_buttons_state("disabled")
        self._clear_table()
        self.total_var.set("TỔNG: 0đ")
        self._redraw_current_preview()

    def run_inference(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Đang chạy", "Model đang xử lý ảnh hiện tại. Đợi chạy xong rồi bấm tiếp.")
            return

        if self.selected_image is None:
            self.choose_image()
            return

        if not self.cv_frame_applied or self.aligned_image_path is None or not self.aligned_image_path.exists():
            self.status_var.set("Chưa được chạy model: cần căn ảnh và bấm 'Áp dụng khung CV' trước.")
            messagebox.showwarning("Chưa áp dụng khung CV", "Hãy kéo/zoom để căn ảnh vào khung xanh, sau đó bấm 'Áp dụng khung CV' rồi mới chạy model.")
            self._set_run_buttons_state("disabled")
            return

        if self.use_egg_var.get() and not self.env_config["api_key"]:
            self.status_var.set("Roboflow đang bật nhưng .env chưa có ROBOFLOW_API_KEY. Sẽ chạy CNN và bỏ qua đếm trứng.")

        self._set_run_buttons_state("disabled", text="Đang chạy...")
        self.status_var.set("Đang load model/chạy nhận diện trên ảnh đã căn khung CV...")
        self.total_var.set("TỔNG: đang tính...")
        self._clear_table()

        image_to_infer = self.aligned_image_path
        self.worker = threading.Thread(target=self._worker_infer, args=(image_to_infer,), daemon=True)
        self.worker.start()

    def rotate_left(self) -> None:
        self._rotate_selected_image(90)

    def rotate_right(self) -> None:
        self._rotate_selected_image(-90)

    def reset_rotation(self) -> None:
        if self.selected_image is None:
            return
        self.image_rotation_deg = 0
        self.rotation_var.set("Góc xoay: 0°")
        self._mark_alignment_dirty("Đã reset xoay. Cần áp dụng lại khung CV.")

    def _rotate_selected_image(self, delta_deg: int) -> None:
        if self.selected_image is None:
            messagebox.showinfo("Chưa có ảnh", "Hãy chọn ảnh trước khi xoay.")
            return
        self.image_rotation_deg = (self.image_rotation_deg + delta_deg) % 360
        self.rotation_var.set(f"Góc xoay: {self.image_rotation_deg}°")
        self._mark_alignment_dirty(f"Đã xoay ảnh {self.image_rotation_deg}°. Cần căn và áp dụng lại khung CV.")

    def reset_zoom(self) -> None:
        self._set_zoom(100)

    def zoom_in(self) -> None:
        self._set_zoom(self.zoom_percent + 10)

    def zoom_out(self) -> None:
        self._set_zoom(self.zoom_percent - 10)

    def _on_zoom_slider_change(self, value: float) -> None:
        self._set_zoom(int(round(value)), sync_slider=False)

    def _set_zoom(self, value: int, sync_slider: bool = True, mark_dirty: bool = True) -> None:
        self.zoom_percent = max(60, min(260, int(value)))
        self.zoom_var.set(f"Zoom: {self.zoom_percent}%")
        if sync_slider and hasattr(self, "zoom_slider"):
            self.zoom_slider.set(self.zoom_percent)
        if mark_dirty and self.selected_image is not None:
            self._mark_alignment_dirty("Đã đổi zoom. Cần áp dụng lại khung CV.", redraw=False)
        self._redraw_current_preview()

    def reset_alignment(self) -> None:
        if self.selected_image is None:
            return
        self.preview_mode = "align"
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._set_zoom(100, mark_dirty=False)
        self._mark_alignment_dirty("Đã reset căn khung. Kéo/zoom lại rồi bấm 'Áp dụng khung CV'.")

    def _mark_alignment_dirty(self, message: str, redraw: bool = True) -> None:
        self.preview_mode = "align"
        self.preview_source_path = None
        self.aligned_image_path = None
        self.cv_frame_applied = False
        self.cv_state_var.set("Khung CV: chưa áp dụng")
        self._set_run_buttons_state("disabled")
        self.status_var.set(message)
        if redraw:
            self._redraw_current_preview()

    def _set_run_buttons_state(self, state: str, text: Optional[str] = None) -> None:
        if hasattr(self, "run_button"):
            label = text or "Nhận diện"
            self.run_button.configure(state=state, text=label)
        if hasattr(self, "run_button_action"):
            label2 = text or "Nhận diện ảnh đã căn khung"
            self.run_button_action.configure(state=state, text=label2)

    # =========================
    # Canvas align / drag / export
    # =========================

    def _load_base_pil(self) -> Image.Image:
        if self.selected_image is None:
            raise RuntimeError("Chưa chọn ảnh.")
        img = Image.open(self.selected_image).convert("RGB")
        img = ImageOps.exif_transpose(img)
        if self.image_rotation_deg % 360:
            img = img.rotate(self.image_rotation_deg, expand=True)
        return img

    def _calc_cv_frame_box(self, canvas_w: int, canvas_h: int) -> Tuple[float, float, float, float]:
        margin = 42
        target_w, target_h = TRAY_SIZE
        scale = min((canvas_w - margin * 2) / target_w, (canvas_h - margin * 2) / target_h)
        scale = max(0.1, scale)
        frame_w = target_w * scale
        frame_h = target_h * scale
        frame_x = (canvas_w - frame_w) / 2
        frame_y = (canvas_h - frame_h) / 2
        return frame_x, frame_y, frame_w, frame_h

    def _get_alignment_geometry(self, img: Image.Image) -> Dict[str, float]:
        canvas_w = max(120, self.preview_canvas.winfo_width())
        canvas_h = max(120, self.preview_canvas.winfo_height())
        fx, fy, fw, fh = self._calc_cv_frame_box(canvas_w, canvas_h)
        fit_scale = min(fw / img.width, fh / img.height)
        display_scale = max(0.01, fit_scale * self.zoom_percent / 100.0)
        iw = img.width * display_scale
        ih = img.height * display_scale
        ix = fx + fw / 2 - iw / 2 + self.pan_x
        iy = fy + fh / 2 - ih / 2 + self.pan_y
        return {
            "canvas_w": float(canvas_w),
            "canvas_h": float(canvas_h),
            "frame_x": float(fx),
            "frame_y": float(fy),
            "frame_w": float(fw),
            "frame_h": float(fh),
            "img_x": float(ix),
            "img_y": float(iy),
            "img_w": float(iw),
            "img_h": float(ih),
            "display_scale": float(display_scale),
        }

    def _frame_is_covered(self, geom: Dict[str, float]) -> bool:
        eps = 1.5
        return (
            geom["img_x"] <= geom["frame_x"] + eps
            and geom["img_y"] <= geom["frame_y"] + eps
            and geom["img_x"] + geom["img_w"] >= geom["frame_x"] + geom["frame_w"] - eps
            and geom["img_y"] + geom["img_h"] >= geom["frame_y"] + geom["frame_h"] - eps
        )

    def apply_cv_frame(self) -> None:
        if self.selected_image is None:
            messagebox.showinfo("Chưa có ảnh", "Hãy chọn ảnh trước.")
            return

        img = self._load_base_pil()
        geom = self._get_alignment_geometry(img)
        if not self._frame_is_covered(geom):
            self.cv_frame_applied = False
            self._set_run_buttons_state("disabled")
            self.cv_state_var.set("Khung CV: ảnh chưa phủ kín")
            self.status_var.set("Ảnh chưa phủ kín khung CV. Hãy zoom lớn hơn hoặc kéo lại để khung xanh nằm hoàn toàn trên ảnh.")
            messagebox.showwarning(
                "Khung chưa hợp lệ",
                "Ảnh chưa phủ kín toàn bộ khung CV màu xanh. Hãy zoom lớn hơn hoặc kéo ảnh lại rồi bấm áp dụng lần nữa.",
            )
            self._redraw_current_preview()
            return

        target_w, target_h = TRAY_SIZE
        fx, fy, fw, fh = geom["frame_x"], geom["frame_y"], geom["frame_w"], geom["frame_h"]
        ix, iy, display_scale = geom["img_x"], geom["img_y"], geom["display_scale"]
        a = (fw / target_w) / display_scale
        b = 0.0
        c = (fx - ix) / display_scale
        d = 0.0
        e = (fh / target_h) / display_scale
        f = (fy - iy) / display_scale

        aligned = img.transform(
            (target_w, target_h),
            Image.Transform.AFFINE,
            (a, b, c, d, e, f),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0),
        )

        temp_dir = self.project_root / "outputs" / "gui_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        out_path = temp_dir / f"{self.selected_image.stem}_cv_aligned.jpg"
        aligned.save(out_path, quality=95)

        self.aligned_image_path = out_path
        self.cv_frame_applied = True
        self.cv_state_var.set("Khung CV: đã áp dụng ✓")
        self.status_var.set("Đã tạo ảnh đã căn khung CV. Bây giờ mới có thể chạy model.")
        self.preview_mode = "align"
        self._set_run_buttons_state("normal")
        self._redraw_current_preview()

    def _start_pan(self, event: tk.Event) -> None:
        if self.preview_mode != "align" or self.selected_image is None:
            return
        self.drag_start = (int(event.x), int(event.y))
        self.preview_canvas.configure(cursor="fleur")

    def _drag_pan(self, event: tk.Event) -> None:
        if self.drag_start is None or self.preview_mode != "align" or self.selected_image is None:
            return
        x0, y0 = self.drag_start
        dx = int(event.x) - x0
        dy = int(event.y) - y0
        self.drag_start = (int(event.x), int(event.y))
        self.pan_x += dx
        self.pan_y += dy
        self._mark_alignment_dirty("Đã kéo ảnh. Cần áp dụng lại khung CV.", redraw=False)
        self._redraw_current_preview()

    def _stop_pan(self, _event: tk.Event) -> None:
        self.drag_start = None
        self.preview_canvas.configure(cursor="")

    def _mousewheel_zoom(self, event: tk.Event) -> None:
        if self.preview_mode != "align" or self.selected_image is None:
            return
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            self._set_zoom(self.zoom_percent + 5)
        else:
            self._set_zoom(self.zoom_percent - 5)

    # =========================
    # Worker / result handling
    # =========================

    def _worker_infer(self, image_path: Path) -> None:
        try:
            use_egg = bool(self.use_egg_var.get())
            if self.inferencer is None:
                self.inferencer = CanteenInferencer(root=self.project_root, use_egg=use_egg)

            bill = self.inferencer.infer_one_tray(image_path, save=True, show=False)
            annotated_path = self._make_gui_result_image(image_path, bill)
            self.result_queue.put(("success", bill, annotated_path))
        except Exception as exc:
            self.result_queue.put(("error", str(exc), traceback.format_exc()))

    def _poll_result_queue(self) -> None:
        try:
            while True:
                item = self.result_queue.get_nowait()
                if item[0] == "success":
                    _status, bill, annotated_path = item
                    self._render_bill(bill)
                    self.preview_source_path = Path(annotated_path)
                    self.preview_mode = "result"
                    self._redraw_current_preview()
                    bill_stem = Path(bill["image_path"]).stem
                    rf_state = "ON" if self.use_egg_var.get() and self.env_config["api_key"] else "OFF"
                    self.status_var.set(f"Xong. Roboflow={rf_state}. Hóa đơn lưu ở outputs/bills/{bill_stem}_bill.*")
                    self._set_run_buttons_state("normal")
                elif item[0] == "error":
                    _status, msg, detail = item
                    self.status_var.set("Có lỗi khi chạy nhận diện.")
                    self.total_var.set("TỔNG: lỗi")
                    self._set_run_buttons_state("normal" if self.cv_frame_applied else "disabled")
                    messagebox.showerror("Lỗi", f"{msg}\n\nChi tiết:\n{detail[-3000:]}")
        except queue.Empty:
            pass
        finally:
            self.after(200, self._poll_result_queue)

    def _render_bill(self, bill: Dict[str, Any]) -> None:
        self._clear_table()
        for item in bill.get("items", []):
            self.result_table.insert(
                "",
                "end",
                values=(
                    item.get("slot_vi", item.get("slot", "")),
                    item.get("pred_class", ""),
                    item.get("final_class", ""),
                    f"{float(item.get('confidence', 0)):.3f}",
                    "-",
                    item.get("price_vnd", format_vnd(item.get("price", 0))),
                ),
            )

        egg_info = bill.get("egg_info", {})
        egg_item = bill.get("egg_item", {})
        egg_equiv = float(egg_info.get("egg_equiv", 0) or 0)
        egg_text = "-" if egg_equiv == 0 else f"{egg_equiv:g}"
        self.result_table.insert(
            "",
            "end",
            values=(
                "Toàn ảnh đã căn",
                "Roboflow",
                egg_item.get("final_class", "trung_roboflow"),
                "-",
                egg_text,
                egg_item.get("price_vnd", format_vnd(egg_item.get("price", 0))),
            ),
        )

        total_text = bill.get("total_vnd", format_vnd(0))
        dish_total = bill.get("dish_total_vnd", format_vnd(0))
        egg_total = bill.get("egg_total_vnd", format_vnd(0))
        self.total_var.set(f"TỔNG: {total_text}  |  món: {dish_total}  |  trứng: {egg_total}")

    def _clear_table(self) -> None:
        if not hasattr(self, "result_table"):
            return
        for row in self.result_table.get_children():
            self.result_table.delete(row)

    def _reset_inferencer_when_option_changed(self) -> None:
        self.inferencer = None
        state = "bật" if self.use_egg_var.get() else "tắt"
        self.status_var.set(f"Đã {state} Roboflow. Lần nhận diện tiếp theo sẽ load lại model.")

    # =========================
    # Image rendering
    # =========================

    def _make_gui_result_image(self, image_path: Path, bill: Dict[str, Any]) -> Path:
        original_img = read_image_cv2(image_path)
        orig_h, orig_w = original_img.shape[:2]
        target_w, target_h = TRAY_SIZE
        img = cv2.resize(original_img, (target_w, target_h))

        for item in bill.get("items", []):
            slot_name = item.get("slot", "")
            if slot_name not in COMPARTMENTS:
                continue
            ymin, ymax, xmin, xmax = COMPARTMENTS[slot_name]
            dish = str(item.get("final_class", ""))
            price = str(item.get("price_vnd", ""))
            conf = float(item.get("confidence", 0))
            label = f"{dish} | {price} | {conf:.2f}"

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.rectangle(img, (xmin, max(0, ymin - 24)), (min(target_w, xmin + 360), ymin), (0, 255, 0), -1)
            cv2.putText(
                img,
                label,
                (xmin + 5, max(18, ymin - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        egg_info = bill.get("egg_info", {})
        input_w = float(egg_info.get("input_width", orig_w) or orig_w)
        input_h = float(egg_info.get("input_height", orig_h) or orig_h)
        sx = target_w / input_w
        sy = target_h / input_h

        for pred in egg_info.get("predictions", []):
            cls = str(pred.get("class_normalized", pred.get("class", "egg")))
            conf = float(pred.get("confidence", 0) or 0)
            x = float(pred.get("x", 0) or 0) * sx
            y = float(pred.get("y", 0) or 0) * sy
            w = float(pred.get("width", 0) or 0) * sx
            h = float(pred.get("height", 0) or 0) * sy
            x1 = max(0, int(x - w / 2))
            y1 = max(0, int(y - h / 2))
            x2 = min(target_w - 1, int(x + w / 2))
            y2 = min(target_h - 1, int(y + h / 2))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 128, 255), 2)
            cv2.putText(
                img,
                f"{cls} {conf:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 128, 255),
                1,
                cv2.LINE_AA,
            )

        egg_item = bill.get("egg_item", {})
        summary = (
            f"Eggs: {float(egg_info.get('egg_equiv', 0) or 0):g} equiv | "
            f"billable {int(egg_info.get('billable_eggs', 0) or 0)} | "
            f"{egg_item.get('price_vnd', format_vnd(0))}"
        )
        cv2.rectangle(img, (8, target_h - 34), (min(target_w - 1, 520), target_h - 8), (0, 128, 255), -1)
        cv2.putText(
            img,
            summary,
            (16, target_h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        out_path = self.project_root / "outputs" / "bills" / f"{Path(image_path).stem}_gui_result.jpg"
        write_image_cv2(out_path, img)
        return out_path

    def _draw_empty_preview(self) -> None:
        self.preview_canvas.delete("all")
        w = max(1, self.preview_canvas.winfo_width())
        h = max(1, self.preview_canvas.winfo_height())
        self.preview_canvas.create_text(
            w // 2,
            h // 2,
            text="Chưa có ảnh",
            fill="#A0A7B5",
            font=("Segoe UI", 16, "bold"),
        )

    def _redraw_current_preview(self) -> None:
        if not hasattr(self, "preview_canvas"):
            return

        if self.preview_mode == "align" and self.selected_image is not None:
            self._redraw_alignment_preview()
            return

        if self.preview_mode == "result" and self.preview_source_path:
            self._redraw_result_preview(self.preview_source_path)
            return

        self._draw_empty_preview()

    def _redraw_alignment_preview(self) -> None:
        try:
            img = self._load_base_pil()
            geom = self._get_alignment_geometry(img)
            self.last_align_geometry = geom

            display_size = (max(1, int(geom["img_w"])), max(1, int(geom["img_h"])))
            display_img = img.resize(display_size, Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(display_img)

            c = self.preview_canvas
            c.delete("all")
            c.create_image(geom["img_x"], geom["img_y"], image=self.preview_photo, anchor="nw")

            fx, fy, fw, fh = geom["frame_x"], geom["frame_y"], geom["frame_w"], geom["frame_h"]
            c.create_rectangle(fx, fy, fx + fw, fy + fh, outline="#00ff66", width=3)

            sx = fw / TRAY_SIZE[0]
            sy = fh / TRAY_SIZE[1]
            for slot_name, box in COMPARTMENTS.items():
                ymin, ymax, xmin, xmax = box
                x1 = fx + xmin * sx
                y1 = fy + ymin * sy
                x2 = fx + xmax * sx
                y2 = fy + ymax * sy
                c.create_rectangle(x1, y1, x2, y2, outline="#00ff66", width=2)
                c.create_text(
                    x1 + 6,
                    max(fy + 12, y1 - 9),
                    text=slot_name,
                    fill="#00ff66",
                    anchor="w",
                    font=("Segoe UI", 9, "bold"),
                )

            covered = self._frame_is_covered(geom)
            state_text = "Đã áp dụng ✓" if self.cv_frame_applied else ("hợp lệ, bấm áp dụng" if covered else "ảnh chưa phủ kín")
            fill = "#00ff66" if covered else "#ffcc00"
            c.create_text(
                fx + fw / 2,
                max(16, fy - 22),
                text=f"KHUNG CV 800x600 - {state_text} | kéo ảnh bằng chuột, cuộn để zoom",
                fill=fill,
                font=("Segoe UI", 12, "bold"),
            )

            if not covered:
                c.create_text(
                    fx + fw / 2,
                    fy + fh + 20,
                    text="Ảnh phải phủ kín toàn bộ khung xanh thì mới được áp dụng/chạy model.",
                    fill="#ffcc00",
                    font=("Segoe UI", 11, "bold"),
                )
        except Exception as exc:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(
                self.preview_canvas.winfo_width() // 2,
                self.preview_canvas.winfo_height() // 2,
                text=f"Không hiển thị được ảnh:\n{exc}",
                fill="#ff6b6b",
                font=("Segoe UI", 12),
            )

    def _redraw_result_preview(self, image_path: Path) -> None:
        try:
            img = Image.open(image_path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            canvas_w = max(120, self.preview_canvas.winfo_width())
            canvas_h = max(120, self.preview_canvas.winfo_height())
            fit_ratio = min((canvas_w - 28) / img.width, (canvas_h - 28) / img.height)
            scale = max(0.05, fit_ratio)
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            self.preview_photo = ImageTk.PhotoImage(img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.preview_photo, anchor="center")
        except Exception as exc:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(
                self.preview_canvas.winfo_width() // 2,
                self.preview_canvas.winfo_height() // 2,
                text=f"Không hiển thị được ảnh kết quả:\n{exc}",
                fill="#ff6b6b",
                font=("Segoe UI", 12),
            )

    # =========================
    # Open helpers
    # =========================

    def open_path(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            if path.name == ".env":
                path.write_text(
                    "ROBOFLOW_API_KEY=\n"
                    f"ROBOFLOW_MODEL_ID={DEFAULT_MODEL_ID}\n"
                    f"ROBOFLOW_API_URL={DEFAULT_API_URL}\n",
                    encoding="utf-8",
                )
                self.reload_env()
            else:
                messagebox.showerror("Không thấy file", f"Không thấy: {path}")
                return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Không mở được", str(exc))

    def open_bills_folder(self) -> None:
        folder = self.project_root / "outputs" / "bills"
        folder.mkdir(parents=True, exist_ok=True)
        self.open_path(folder)

    def open_menu_file(self) -> None:
        self.open_path(self.project_root / "models" / "cnn" / "menu.json")


def main() -> None:
    app = CanteenGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
