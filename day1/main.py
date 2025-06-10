import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import math


def percentile_stretch_large(tensor, q_min=2, q_max=98, chunk_size=2048):


    all_quantiles = []


    q_values = [q_min / 100, q_max / 100]


    num_chunks_h = math.ceil(tensor.shape[1] / chunk_size)
    num_chunks_w = math.ceil(tensor.shape[2] / chunk_size)

    for i in range(num_chunks_h):
        for j in range(num_chunks_w):
            h_start = i * chunk_size
            h_end = min((i + 1) * chunk_size, tensor.shape[1])
            w_start = j * chunk_size
            w_end = min((j + 1) * chunk_size, tensor.shape[2])

            chunk = tensor[:, h_start:h_end, w_start:w_end]

            # 分别计算每个分位数（兼容新版PyTorch）
            chunk_lower = torch.quantile(chunk, q_values[0], dim=1).min(dim=1).values
            chunk_upper = torch.quantile(chunk, q_values[1], dim=1).max(dim=1).values
            chunk_quantiles = torch.stack([chunk_lower, chunk_upper], dim=1)

            all_quantiles.append(chunk_quantiles)

    # 合并所有块的统计结果
    stacked_quantiles = torch.stack(all_quantiles)
    lower = stacked_quantiles[:, :, 0].min(dim=0).values
    upper = stacked_quantiles[:, :, 1].max(dim=0).values

    # 应用拉伸
    stretched = (tensor - lower.view(-1, 1, 1)) / (upper - lower).view(-1, 1, 1) * 255
    return torch.clamp(stretched, 0, 255).to(torch.uint8)


def sentinel2_to_rgb(input_path, output_path):
    try:
        # 1. 读取数据
        with rasterio.open(input_path) as src:
            bands = src.read()
            print(f"原始数据形状: {bands.shape} (波段, 高, 宽)")
            print(f"原始值范围: {bands.min()} - {bands.max()}")

        # 2. 转换为PyTorch张量
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor = torch.from_numpy(bands.astype(np.float32)).to(device)

        # 3. 分块归一化
        stretched = percentile_stretch_large(tensor, chunk_size=1024)  # 减小分块大小节省内存

        # 4. 提取RGB波段（假设顺序是B2/B3/B4/B8/B12）
        rgb_indices = [2, 1, 0]  # B4(R), B3(G), B2(B)
        rgb_tensor = stretched[rgb_indices]

        # 5. 保存结果
        rgb_np = rgb_tensor.cpu().permute(1, 2, 0).numpy()
        plt.imsave(output_path, rgb_np)

        # 6. 显示结果（缩小显示尺寸）
        plt.figure(figsize=(12, 6))

        # 显示原始波段（缩小尺寸）
        display_band = bands[0][::10, ::10]  # 缩小10倍显示
        plt.subplot(1, 2, 1)
        plt.title("原始波段1(缩小显示)")
        plt.imshow(display_band, cmap='gray', vmin=0, vmax=10000)

        # 显示RGB结果
        plt.subplot(1, 2, 2)
        plt.title("RGB合成")
        plt.imshow(rgb_np[::10, ::10, :])  # 缩小10倍显示
        plt.show()

        print(f"处理完成！结果已保存到: {output_path}")

    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise


if __name__ == "__main__":
    input_tif = "2019_1101_nofire_B2348_B12_10m_roi.tif"
    output_jpg = "sentinel2_rgb_output.jpg"
    sentinel2_to_rgb(input_tif, output_jpg)