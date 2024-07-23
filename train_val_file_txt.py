import os

# Định nghĩa các đường dẫn
base_path = '/data/Phat/VinAI/subdata_mvsgaussian_depth_route1_route2_p2'
train_path = os.path.join(base_path, 'train')
test_path = os.path.join(base_path, 'test')

output_train_file = '/data/Phat/MVSGaussian/data/mvsgs/dtu_train_all.txt'
output_test_file = '/data/Phat/MVSGaussian/data/mvsgs/dtu_val_all.txt'

# Hàm để ghi danh sách các thư mục con vào file
def write_subfolders_to_file(directory, output_file):
    with open(output_file, 'w') as f:
        for subfolder in os.listdir(directory):
            subfolder_path = os.path.join(directory, subfolder)
            if os.path.isdir(subfolder_path):
                f.write(subfolder + '\n')

# Ghi danh sách thư mục con từ thư mục train vào file
write_subfolders_to_file(train_path, output_train_file)

# Ghi danh sách thư mục con từ thư mục test vào file
write_subfolders_to_file(test_path, output_test_file)

print("Đã ghi danh sách thư mục con vào các file tương ứng.")
