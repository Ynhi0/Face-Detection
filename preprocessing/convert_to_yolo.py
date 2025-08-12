import os
import cv2
from tqdm import tqdm

def process_wider_to_yolo_format(annotation_file, images_dir, output_labels_dir, output_img_dir):
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)

    with open(annotation_file, 'r',encoding='utf-8') as f:
        lines = f.readlines()

    file_count = 0
    i = 0
    while i < len(lines):
        img_path_relative = lines[i].strip()
        i += 1
        num_boxes_str = lines[i].strip()
        i += 1

        try:
            num_boxes = int(num_boxes_str)
        except (ValueError, IndexError):
            continue

        full_src_img_path = os.path.join(images_dir, img_path_relative)
        img_name = os.path.basename(img_path_relative)
        
        # Xử lý trường hợp ảnh không có khuôn mặt nào
        if num_boxes == 0:
            i += 1  # Bỏ qua dòng bounding box không hợp lệ
            if not os.path.exists(full_src_img_path):
                continue
            
            # Vẫn copy ảnh và tạo file label rỗng
            label_name = os.path.splitext(img_name)[0] + '.txt'
            open(os.path.join(output_labels_dir, label_name), 'w').close()
            dest_img_path = os.path.join(output_img_dir, img_name)
            if not os.path.exists(dest_img_path):
                cv2.imwrite(dest_img_path, cv2.imread(full_src_img_path))
            file_count += 1
            continue

        if not os.path.exists(full_src_img_path):
            i += num_boxes
            continue

        img = cv2.imread(full_src_img_path)
        if img is None:
            i += num_boxes
            continue
        
        img_height, img_width, _ = img.shape
        
        # Copy ảnh sang thư mục đích
        dest_img_path = os.path.join(output_img_dir, img_name)
        if not os.path.exists(dest_img_path):
             cv2.imwrite(dest_img_path, img)

        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(output_labels_dir, label_name)
        
        yolo_annotations = []
        for j in range(num_boxes):
            bbox_line = lines[i + j].strip().split()
            if len(bbox_line) < 4:
                continue

            x1, y1, w, h = [int(v) for v in bbox_line[:4]]

            if w <= 0 or h <= 0:
                continue

            x_center = (x1 + w / 2.0) / img_width
            y_center = (y1 + h / 2.0) / img_height
            width_norm = w / img_width
            height_norm = h / img_height
            
            yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")

        with open(label_path, 'w') as label_file:
            label_file.write("\n".join(yolo_annotations))

        i += num_boxes
        file_count += 1
    
    return file_count