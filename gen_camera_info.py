import json
import numpy as np
import os

def scale_matrix(matrix, x_scale, y_scale):
    scaled_matrix = matrix.copy()
    scaled_matrix[0, 3] *= x_scale
    scaled_matrix[1, 3] *= y_scale
    return scaled_matrix

def save_matrix_to_file(matrix, filename, intrinsic_matrix, depth_range, output_dir):
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write('extrinsic\n')
        np.savetxt(f, matrix, fmt='%.10e', delimiter=' ')
        f.write('\nintrinsic\n')
        np.savetxt(f, intrinsic_matrix, fmt='%.1f', delimiter=' ')
        f.write('\n')
        f.write(f'{depth_range[0]} {depth_range[1]}\n')

def main(json_file, output_dir):
    # Đọc file JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Định nghĩa ma trận nội suy (intrinsic) và dải độ sâu (depth range)
    intrinsic_matrix = np.array([
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 256.0],
        [0.0, 0.0, 1.0]
    ])
    depth_range = [425.0, 905.0]

    # Tạo thư mục lưu file nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Xử lý từng ma trận
    for key, matrix_list in data['transform'].items():
        matrix = np.array(matrix_list)
        
        # Scale ma trận
        scaled_matrix = scale_matrix(matrix, 640 / 320, 512 / 200)
        
        # Chuyển ma trận từ cam2world sang world2cam
        world2cam_matrix = np.linalg.inv(scaled_matrix)
        
        # Lưu ma trận vào file
        filename = key + '.txt'
        save_matrix_to_file(world2cam_matrix, filename, intrinsic_matrix, depth_range, output_dir)

    print("Đã hoàn thành việc lưu các ma trận vào các file.")

# Đường dẫn đến file JSON và thư mục đầu ra
json_file = '/data/Phat/mvsplat_config/transforms_ego_train_final.json'  # Thay đổi đường dẫn này
output_dir = '/data/Phat/VinAI/subdata_mvsgaussian_depth_route1_route2_p2/camera'  # Thay đổi đường dẫn này

main(json_file, output_dir)
