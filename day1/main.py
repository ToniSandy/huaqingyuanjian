import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import os


def process_sentinel2_image(input_path, output_path):

    try:

        with rasterio.open(input_path) as src:

            bands = src.read()
            profile = src.profile

            print(f"输入图像信息:")
            print(f"  波段数: {src.count}")
            print(f"  宽度: {src.width}, 高度: {src.height}")
            print(f"  数据类型: {src.dtypes[0]}")
            print(f"  范围: {bands.min()} - {bands.max()}")


            stretched_bands = []
            for band in bands:
                p2, p98 = np.percentile(band[band > 0], (2, 98))  # 忽略0值
                band_stretched = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
                stretched_bands.append(band_stretched)

            stretched_bands = np.array(stretched_bands).astype(np.uint8)

            print("\n拉伸后数据范围:")
            print(f"  Min: {stretched_bands.min()}, Max: {stretched_bands.max()}")

            rgb_image = np.stack([
                stretched_bands[2],  # 红色 (B4)
                stretched_bands[1],  # 绿色 (B3)
                stretched_bands[0]  # 蓝色 (B2)
            ], axis=0)

            # 更新元数据
            profile.update(
                count=3,
                dtype='uint8',
                driver='GTiff'
            )

            # 4. 保存结果
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(rgb_image)

            print(f"\n处理完成! 结果已保存到: {output_path}")

            # 显示原始图像和结果图像
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            # 显示第一个波段
            show(src, cmap='gray')
            plt.title('原始图像 (第一个波段)')

            plt.subplot(1, 2, 2)
            # 正确格式化RGB图像为(高度, 宽度, 波段)
            rgb_display = np.moveaxis(rgb_image, 0, -1)
            plt.imshow(rgb_display)
            plt.title('RGB图像')

            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"处理图像时出错: {e}")
        raise  # 重新抛出异常以便调试


if __name__ == "__main__":

    input_filename = "2019_1101_nofire_B2348_B12_10m_roi.tif"
    input_path = os.path.join(os.path.dirname(__file__), input_filename)


    output_filename = "2019_1101_nofire_RGB_8bit.tif"
    output_path = os.path.join(os.path.dirname(__file__), output_filename)


    process_sentinel2_image(input_path, output_path)