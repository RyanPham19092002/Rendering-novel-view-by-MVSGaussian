from PIL import Image

# Đường dẫn tới file ảnh
image_path = '/data/Phat/VinAI/mvs_training/dtu/Rectified/scan1_train/rect_001_0_r5000.png'

# Mở ảnh
with Image.open(image_path) as img:
    # Lấy kích thước ảnh
    width, height = img.size
    print(f"Kích thước ảnh: {width} x {height}")
