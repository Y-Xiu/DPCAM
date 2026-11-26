import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import random

CROP_SIZE = 1024

CONFIG_2 = {
    'input_dir': './pcbdata1/dataset2',
    'output_dir': './Processed_Dataset2',
    'classes': ['background', 'good', 'exc_solder', 'poor_solder', 'spike'],
}


# ===========================================

def create_crop(image, shapes, center_point, crop_size, save_name, output_dirs, class_map):
    img_h, img_w = image.shape[:2]
    cx, cy = center_point

    jitter = 50
    x1 = int(cx + random.randint(-jitter, jitter) - crop_size / 2)
    y1 = int(cy + random.randint(-jitter, jitter) - crop_size / 2)

    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    if x2 > img_w:
        x2 = img_w
        x1 = img_w - crop_size
    if y2 > img_h:
        y2 = img_h
        y1 = img_h - crop_size

    crop_img = image[y1:y2, x1:x2]

    crop_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)

    for shape in shapes:
        label = shape['label']
        if label not in class_map: continue

        points = np.array(shape['points'])
        points[:, 0] -= x1
        points[:, 1] -= y1
        points = points.astype(np.int32)

        cv2.fillPoly(crop_mask, [points], class_map[label])

    cv2.imwrite(os.path.join(output_dirs['images'], save_name + '.jpg'), crop_img)
    cv2.imwrite(os.path.join(output_dirs['masks'], save_name + '.png'), crop_mask)


def process_dataset(config):
    input_dir = config['input_dir']
    output_dir = config['output_dir']
    classes = config['classes']
    multipliers = config['multiplier']


    class_map = {label: i for i, label in enumerate(classes)}

    img_out = os.path.join(output_dir, 'images')
    mask_out = os.path.join(output_dir, 'masks')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    print(f"\n正在处理: {input_dir} -> {output_dir}")
    print(f"类别: {class_map}")

    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    count = 0
    for json_file in tqdm(json_files):
        json_path = os.path.join(input_dir, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue

        # 找对应的图片
        img_name = os.path.splitext(json_file)[0] + '.jpg'
        img_path = os.path.join(input_dir, img_name)
        if not os.path.exists(img_path): continue

        image = cv2.imread(img_path)
        if image is None: continue

        for i, shape in enumerate(data['shapes']):
            label = shape['label']

            if label not in class_map:
                continue

            multiplier = multipliers.get(label, 1)
            points = np.array(shape['points'])
            cx = int(np.mean(points[:, 0]))
            cy = int(np.mean(points[:, 1]))

            for m in range(multiplier):
                save_name = f"{os.path.splitext(json_file)[0]}_{i}_{label}_{m}"
                create_crop(image, data['shapes'], (cx, cy), CROP_SIZE, save_name,
                            {'images': img_out, 'masks': mask_out}, class_map)
                count += 1
    print(f"生成了 {count} 张训练图。")


if __name__ == '__main__':

    if os.path.exists(CONFIG_2['input_dir']):
         process_dataset(CONFIG_2)
    else:
         print(f"跳过 Dataset 2，因为找不到文件夹 {CONFIG_2['input_dir']}")