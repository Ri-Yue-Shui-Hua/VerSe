# -*- coding : UTF-8 -*-
# @file   : reorient_image.py
# @Time   : 2024-03-28 16:45
# @Author : wmz
from preprocess import *


def get_files(path, suffix):
    return [
        os.path.join(root, file)
        for root, dirs, files in os.walk(path)
        for file in files
        if file.endswith(suffix)
    ]


def reorient_data(file_folder, suffix, dst_folder):
    # 数据方向重置成LPS
    file_list = get_files(file_folder, suffix)
    for i, file in enumerate(file_list):
        print("processing ", i, " of ", len(file_list), " files ......")
        image = nib.load(file)
        new_img = ct_reorient(image)
        file_id = os.path.basename(file)
        dst_file = os.path.join(dst_folder, file_id)
        nib.save(new_img, dst_file)


def resample_to_spacing(file_folder, suffix, dst_folder):
    # 数据重采样到spacing=1mm
    file_list = get_files(file_folder, suffix)
    for i, file in enumerate(file_list):
        print("processing ", i, " of ", len(file_list), " files ......")
        image = nib.load(file)
        if "_seg" in file:
            new_img = ct_resample(image, is_label=True)
        else:
            new_img = ct_resample(image)
        file_id = os.path.basename(file)
        dst_file = os.path.join(dst_folder, file_id)
        nib.save(new_img, dst_file)


def generate_labels_annotates(file_folder, suffix, reorient_folder, spacing1_folder):
    # 生成质心标签
    # 质心需要重置方向，适应重采样到spacing=1mm的图像
    # img_file_list = get_files(reorient_folder, ".nii.gz")
    file_list = get_files(file_folder, suffix)
    for i, file in enumerate(file_list):
        print("processing ", i, " of ", len(file_list), " files ......")
        file_id = os.path.basename(file).replace(suffix, "")
        img_file = os.path.join(
            reorient_folder, file_id.replace("subreg_ctd", "vert_msk.nii.gz")
        )
        output_path = os.path.join(
            reorient_folder, file_id.replace("subreg_ctd", "subreg_ctd.json")
        )
        spacing1_output_path = os.path.join(
            spacing1_folder, file_id.replace("subreg_ctd", "subreg_ctd.json")
        )
        image = nib.load(img_file)
        ctd_list = get_json(file)
        # dict_list = centroids_to_dict(ctd_list)
        out_list = centroids_reorient(ctd_list, image, decimals=1)  # 方向转正
        # centroids_save(out_list, output_path)
        spacing1_list = centroids_rescale(
            out_list, image, voxel_spacing=(1, 1, 1)
        )  # 重采样到spacing=1mm
        centroids_save(spacing1_list, spacing1_output_path)


if __name__ == "__main__":
    ct_file_folder = r"E:/Dataset/VerSe19/dataset-verse19test/rawdata"
    label_file_folder = r"E:/Dataset/VerSe19/dataset-verse19test/derivatives"
    reorient_folder = r"E:/Dataset/VerSe19/dataset-verse19test/oriented"
    spacing1_folder = r"E:/Dataset/VerSe19/dataset-verse19test/1mm"
    # drr_folder = "E:/Data/VerSe/dataset-verse19validation/DRR/"
    # heatmap_folder = "E:/Data/VerSe/dataset-verse19validation/heatmap/"
    if not os.path.exists(reorient_folder):
        os.makedirs(reorient_folder)
    if not os.path.exists(spacing1_folder):
        os.makedirs(spacing1_folder)

    suffix = ".nii.gz"
    json_suffix = ".json"
    # 1. 数据方向重置成LPS
    # reorient_data(ct_file_folder, suffix, reorient_folder)
    # reorient_data(label_file_folder, suffix, reorient_folder)
    # # 2. 数据重采样到spacing=1mm
    # resample_to_spacing(reorient_folder, suffix, spacing1_folder)
    # 3. 质心点调整
    generate_labels_annotates(
        label_file_folder, json_suffix, reorient_folder, spacing1_folder
    )
