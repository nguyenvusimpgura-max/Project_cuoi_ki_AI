# Canteen Project - bản chạy trên VSCode

Bản này đã được chuyển từ notebook Google Colab sang project local để chạy trong VSCode/terminal, kèm thêm GUI desktop `gui.py`.

## 1. Mở project trong VSCode

Mở đúng thư mục này:

```bash
canteen_project_vscode
```

Cấu trúc quan trọng:

```text
main.py                         # chạy bằng terminal/CLI
gui.py                          # giao diện desktop
requirements.txt
.env.example
models/cnn/final_canteen_cnn.keras
models/cnn/class_names.json
models/cnn/menu.json            # chỉnh giá món ở đây
data/test_trays/
data/raw_trays/
inference/
outputs/
```

## 2. Tạo môi trường Python

Nên dùng Python 3.10, 3.11 hoặc 3.12.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Windows thường có sẵn `tkinter` để chạy GUI. Nếu dùng Linux mà báo thiếu tkinter, cài thêm `python3-tk` bằng package manager của hệ điều hành.

## 3. Chạy GUI

Cách nhanh nhất:

```bash
python gui.py
```

Hoặc trên Windows có thể double-click/chạy:

```bat
run_gui.bat
```

Trong VSCode: vào **Run and Debug** → chọn **Run GUI**.

GUI có các phần chính:

- **Chọn ảnh**: chọn ảnh khay cần nhận diện.
- **Bật Roboflow đếm trứng**: chỉ bật khi đã có `.env` chứa API key. Trong GUI, Roboflow sẽ đếm trên toàn bộ ảnh **đã căn khung CV** trước khi CNN crop/resize từng ô.
- **Kéo/zoom căn khung CV**: sau khi chọn ảnh, kéo ảnh bằng chuột và chỉnh zoom sao cho khay phủ kín khung xanh 800x600.
- **Áp dụng khung CV**: bắt buộc bấm nút này để tạo ảnh `outputs/gui_temp/*_cv_aligned.jpg`; nếu chưa áp dụng thì nút chạy model bị khóa.
- **Nhận diện ảnh đã căn khung**: chỉ chạy sau khi khung CV hợp lệ và đã áp dụng.
- **Bảng kết quả món + trứng riêng**: hiện món từng ô và thêm một dòng trứng Roboflow toàn ảnh.
- **Tổng tiền**: hiện tổng hóa đơn.
- **Mở file chỉnh giá menu**: mở `models/cnn/menu.json` để sửa giá món.
- **Mở thư mục hóa đơn**: mở `outputs/bills`.


### GUI bản đã cải tổ

- Preview ảnh dùng canvas riêng, không còn lỗi `image_canvas`/`zoom_label_var`.
- Có khung CV màu xanh gồm toàn khay 800x600 và 5 ô crop cố định.
- Có kéo-thả ảnh trực tiếp trên preview, cuộn chuột để zoom nhanh, slider để zoom thủ công.
- Nút **Nhận diện** bị khóa cho tới khi bấm **Áp dụng khung CV** thành công.
- Có nút **Reload .env** và **Mở .env**.
- Bảng kết quả tách rõ CNN dự đoán, món tính tiền, confidence, trứng và giá.
- `thit_kho_trung` được gộp về `thit_kho`; trứng Roboflow được tính thành dòng riêng, không còn gắn vào ô thịt kho.

## 4. Cấu hình Roboflow để đếm trứng

Bản nộp **không kèm `.env` thật** để tránh lộ API key. Muốn bật Roboflow, copy `.env.example` thành `.env`:

```env
ROBOFLOW_API_KEY=...
ROBOFLOW_MODEL_ID=egg2-3omrr/1
ROBOFLOW_API_URL=https://serverless.roboflow.com
```

GUI sẽ tự đọc `.env`. Nếu thấy API key, checkbox **Bật Roboflow đếm trứng toàn ảnh đã căn khung** sẽ tự bật. Nếu chưa có key, chương trình vẫn chạy được phần CNN/OpenCV và có thể tắt Roboflow.

Nếu chưa cần đếm trứng, tắt checkbox **Bật Roboflow đếm trứng toàn ảnh đã căn khung** trong GUI, hoặc chạy CLI với `--no-egg`.

Không upload/chia sẻ `.env` thật nếu project đưa lên GitHub hoặc nộp nơi công khai. Khi cần demo trực tiếp, tạo `.env` cục bộ trên máy đang chạy.

## 5. Chỉnh giá món

Giá nằm trong:

```text
models/cnn/menu.json
```

Ví dụ:

```json
{
  "com": 10000,
  "thit_kho": 25000,
  "rau_xao": 10000
}
```

Sau khi sửa giá, lưu file rồi chạy nhận diện lại. Nếu GUI đang mở và bạn vừa sửa `menu.json`, hãy tắt GUI mở lại để chắc chắn model đọc giá mới.

## 6. Chạy bằng terminal/CLI

Chạy 1 ảnh:

```bash
python main.py --image data/test_trays/test002.jpg
```

Hoặc ảnh còn lại trong bộ test:

```bash
python main.py --image "data/test_trays/test_001.jpg.png"
```

Chạy toàn bộ thư mục test:

```bash
python main.py --input-dir data/test_trays
```

Chỉ dùng CNN, không gọi Roboflow:

```bash
python main.py --input-dir data/test_trays --no-egg
```

## 7. Kết quả lưu ở đâu?

Sau khi chạy, project sẽ tự tạo:

```text
outputs/gui_temp/*_cv_aligned.jpg  # ảnh đã căn khung từ GUI
outputs/bills/                 # bill .json, .txt, ảnh kết quả GUI
outputs/logs/inference_summary.csv
inference/cropped_slots/        # crop từng ô
inference/cropped_slots_224/    # crop resize 224x224
inference/debug_boxes/          # ảnh khay có vẽ box
inference/egg_debug/            # debug Roboflow nếu bật đếm trứng
```

## 8. Lưu ý logic trứng

Logic mới trong `main.py`:

```text
1. Đọc ảnh đầu vào.
2. Gọi Roboflow đếm trứng trên toàn bộ ảnh đầu vào, chưa resize trong main.py.
3. Sau đó mới resize ảnh về 800x600 và crop từng ô cho CNN.
4. CNN chỉ phân loại món từng ô.
5. `thit_kho_trung` được gộp thành `thit_kho`.
6. Trứng được tính riêng bằng dòng `trung_roboflow`.
```

Riêng khi chạy bằng GUI, ảnh đầu vào của `main.py` là ảnh đã được bạn kéo/zoom và bấm **Áp dụng khung CV**. Vì vậy crop CNN sẽ khớp khung xanh thay vì resize bừa theo ảnh gốc.

Giá trứng riêng dùng hằng số `EXTRA_EGG_PRICE = 6000` trong `main.py`. Số trứng quy đổi được tính bằng:

```text
egg_equiv = egg_whole + 0.5 * egg_half
billable_eggs = ceil(egg_equiv)
```

Nếu Roboflow không chạy vì thiếu `.env` hoặc bạn tắt Roboflow trong GUI, `egg_equiv = 0` và dòng trứng sẽ có giá `0đ`.

## 9. Chạy bằng nút Run/Debug trong VSCode

Project đã có `.vscode/launch.json`. Vào tab **Run and Debug**, chọn một trong các cấu hình:

- `Run GUI`
- `Run one test tray`
- `Run all test trays`
- `Run all test trays - no egg`

## 10. Nếu TensorFlow lỗi khi cài

Thử cập nhật pip trước:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Nếu vẫn lỗi, kiểm tra VSCode đã chọn đúng Python interpreter trong `.venv` chưa: `Ctrl + Shift + P` → `Python: Select Interpreter` → chọn `.venv`.
