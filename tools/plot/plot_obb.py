import os
import cv2
import numpy as np
from tqdm import tqdm

def draw_rotated_box(image, points, color, label):
    """在图像上绘制旋转框"""
    points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
    # 在框的左上角绘制标签
    cv2.putText(image, label, (points[0][0][0], points[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def visualize_annotations(image_dir, annotation_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取图像文件和标注文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.txt')]

    # 确保图像文件和标注文件一一对应
    image_files.sort()
    annotation_files.sort()

    for image_file, annotation_file in tqdm(zip(image_files, annotation_files), total=len(image_files)):
        image_path = os.path.join(image_dir, image_file)
        annotation_path = os.path.join(annotation_dir, annotation_file)

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f'Failed to read image: {image_path}')
            continue

        # 读取标注文件
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            coords = list(map(int, parts[:8]))
            label = parts[8]
            truncated = parts[9]

            # 绘制旋转框
            color = (0, 255, 0)  # 绿色
            draw_rotated_box(image, coords, color, label)

        # 保存带有旋转框的图像
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, image)

# 示例用法
# image_dir = '/opt/data/private/fcf/MFOD_master/data/clip_dronevehicle/test/rgb/images'
# annotation_dir = '/opt/data/private/fcf/MFOD_master/data/clip_dronevehicle/test/labels'
# output_dir = '/opt/data/private/fcf/MFOD_master/tools/dataset_vis/clip_dronevehicle_test_rgb'
# visualize_annotations(image_dir, annotation_dir, output_dir)

image_dir = '/opt/data/private/fcf/MFOD_master/data/clip_dronevehicle/train/rgb/images'
annotation_dir = '/opt/data/private/fcf/MFOD_master/data/clip_dronevehicle/train/labels'
output_dir = '/opt/data/private/fcf/MFOD_master/tools/dataset_vis/clip_dronevehicle_train_rgb'
visualize_annotations(image_dir, annotation_dir, output_dir)