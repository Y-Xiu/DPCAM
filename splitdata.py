import os
import shutil
import random
from glob import glob
import pandas as pd


img_dir = "data1/images"
mask_dir = "data1/masks"
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

splits = ["train", "val", "test"]
split_ratios = [0.7, 0.15, 0.15]

classes = ["good", "exc_solder", "poor_solder", "spike"]


mask_paths = glob(os.path.join(mask_dir, "*.png"))
data = []

for mask_path in mask_paths:
    mask_name = os.path.basename(mask_path)
    cls = None
    for c in classes:
        if f"_{c}_" in mask_name:
            cls = c
            break
    if cls is None:
        print(f"WARNING: 未识别类别: {mask_name}")
        continue

    img_name = os.path.splitext(mask_name)[0] + ".jpg"
    img_path = os.path.join(img_dir, img_name)
    if not os.path.exists(img_path):
        print(f"WARNING: 对应 image 不存在: {img_path}")
        continue
    data.append([img_path, mask_path, cls])

df = pd.DataFrame(data, columns=["img", "mask", "class"])
print("数据总量:", len(df))
print(df["class"].value_counts())


for s in splits:
    os.makedirs(os.path.join(output_dir, s, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, s, "masks"), exist_ok=True)


train_files, val_files, test_files = [], [], []

for cls in classes:
    cls_df = df[df["class"] == cls]
    cls_list = cls_df.to_dict(orient="records")
    random.shuffle(cls_list)
    n = len(cls_list)
    n_train = int(split_ratios[0] * n)
    n_val = int(split_ratios[1] * n)
    n_test = n - n_train - n_val

    train_files += cls_list[:n_train]
    val_files += cls_list[n_train:n_train+n_val]
    test_files += cls_list[n_train+n_val:]

print("训练集数量:", len(train_files))
print("验证集数量:", len(val_files))
print("测试集数量:", len(test_files))


def copy_files(file_list, split):
    for item in file_list:
        shutil.copy(item["img"], os.path.join(output_dir, split, "images", os.path.basename(item["img"])))
        shutil.copy(item["mask"], os.path.join(output_dir, split, "masks", os.path.basename(item["mask"])))

copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("数据集划分完成！")