# cv_project (OpenCV - Computer Vision, no training)

## Yêu cầu
- Windows 10/11
- Python **3.11 hoặc 3.12, 3.13. Không được cài 3.14 vì opencv chưa hỗ trợ** 
- Text editor hỗ trợ (có thể không cài, dùng python là đủ): VSCode + Extension: **Python** (Microsoft), **Pylance**

---

## 1) Clone / tải project về
Mở CMD và vào thư mục muốn đặt project, ví dụ:
```bat
cd /d C:\dev
git clone https://github.com/yourusername/computerVision.git
cd cv_project
```
## 2) Tạo & kích hoạt môi trường ảo (venv)
Trong thư mục project (`cv_project`):

```bat
Kiểm tra version: python --version
Tạo môi trường ảo: python -m venv .venv
Kích hoạt môi trường ảo: .venv\Scripts\activate
Khi kích hoạt nó phải hiện (.venv) ở đầu dòng lệnh
```
## 3) Cài đặt thư viện cần thiết
```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
## 4) Kiểm tra
```bat  
python -c "import cv2; import numpy as np; print('OpenCV:', cv2.__version__); print('Numpy:', np.__version__)"
```
## 5) Tắt môi trường ảo
```bat
deactivate
```