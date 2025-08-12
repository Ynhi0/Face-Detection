import os
import sys
import yaml
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# Import từ các module khác trong project
from preprocessing.convert_to_yolo import process_wider_to_yolo_format

def load_config(path):
    with open(path, 'r',encoding = 'utf-8') as f:
        return yaml.safe_load(f)

def create_master_yolo_config(data_cfg, output_dir):
    master_config_path = os.path.join(output_dir, 'master_config.yaml')
    
    # Tạo nội dung cho file YAML
    yolo_config = {
        'path': os.path.abspath(output_dir), # Đường dẫn tuyệt đối đến thư mục yolo_output
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['face']
    }
    
    with open(master_config_path, 'w') as f:
        yaml.dump(yolo_config, f, sort_keys=False)
        
    print(f"Đã tạo file config YOLO chính tại: {master_config_path}")
    return master_config_path

def main():
    print("Bắt đầu quy trình huấn luyện nhận diện khuôn mặt...")
    print("\n[Bước 1/4] Tải file cấu hình...")
    # Đường dẫn đến các file config, tính từ thư mục gốc
    data_cfg_path = os.path.join(PROJECT_ROOT, 'configs/data_config.yaml')
    train_cfg_path = os.path.join(PROJECT_ROOT, 'configs/train_config.yaml')
    
    data_cfg = load_config(data_cfg_path)
    train_cfg = load_config(train_cfg_path)
    
    output_dir = os.path.join(PROJECT_ROOT, data_cfg['yolo_dataset_dir'])
    print("\n[Bước 2/4] Chuẩn bị dữ liệu cho YOLO...")
    
    # Kiểm tra xem dữ liệu đã được xử lý chưa để tiết kiệm thời gian
    if os.path.exists(os.path.join(output_dir, 'labels/val')):
        print("Dữ liệu định dạng YOLO đã tồn tại, bỏ qua bước chuyển đổi.")
    else:
        print("Bắt đầu chuyển đổi dữ liệu WIDER FACE...")
        # Xử lý tập train
        print(" > Đang xử lý tập train...")
        count_train = process_wider_to_yolo_format(
            annotation_file=os.path.join(PROJECT_ROOT, data_cfg['train_annotation_file']),
            images_dir=os.path.join(PROJECT_ROOT, data_cfg['train_images']),
            output_labels_dir=os.path.join(output_dir, 'labels/train'),
            output_img_dir=os.path.join(output_dir, 'images/train')
        )
        print(f" > Hoàn tất xử lý {count_train} ảnh train.")
        
        # Xử lý tập validation
        print(" > Đang xử lý tập validation...")
        count_val = process_wider_to_yolo_format(
            annotation_file=os.path.join(PROJECT_ROOT, data_cfg['val_annotation_file']),
            images_dir=os.path.join(PROJECT_ROOT, data_cfg['val_images']),
            output_labels_dir=os.path.join(output_dir, 'labels/val'),
            output_img_dir=os.path.join(output_dir, 'images/val')
        )
        print(f" > Hoàn tất xử lý {count_val} ảnh validation.")

    print("\n[Bước 3/4] Tạo file config cho YOLO...")
    master_yaml_path = create_master_yolo_config(data_cfg, output_dir)
    
    print("\n[Bước 4/4] Bắt đầu huấn luyện model YOLO...")
    try:
        model = YOLO(train_cfg['model'])
        
        model.train(
            data=master_yaml_path,
            epochs=train_cfg['epochs'],
            imgsz=train_cfg['imgsz'],
            batch=train_cfg['batch'],
            device=train_cfg['device'],
            patience=train_cfg['patience'],
            workers=train_cfg['workers'],
            project=train_cfg['project_name']
        )
        print("\n--- QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT ---")
        
    except Exception as e:
        print(f"\n!!! LỖI: Có lỗi xảy ra trong quá trình huấn luyện: {e}")

if __name__ == '__main__':
    main()