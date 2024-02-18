import tifffile

def split_tif(input_path, output_folder):
    # 读取TIF文件
    with tifffile.TiffFile(input_path) as tif:
        # 获取TIF文件中的每个slice
        slices = len(tif.pages)

        # 循环遍历每个slice
        for i, page in enumerate(tif.pages):
            # 保存当前slice为单独的文件
            output_file = f"{output_folder}/slice_{i + 1}.jpg"
            tifffile.imsave(output_file, page.asarray())

            print(f"Slice {i + 1} saved as {output_file}")

# 用法示例
input_tif_path = "./train/3D_structure.tif"
output_folder_path = "./train/images"

split_tif(input_tif_path, output_folder_path)
