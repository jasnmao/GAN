# CelebA GAN 图像修复

使用条件 GAN 根据 CelebA 人脸图像的可见区域补全缺损。训练时生成器接收被遮挡图像和二值 mask，PatchGAN 判别器区分真实图像与修复结果；损失由缺损区域 L1 重建误差和 hinge 对抗损失组成。可见区域在输出合成时直接复制原像素，避免已知内容被改动。

训练数据使用 CelebA 官方 `list_eval_partition.txt` 划分（train/valid/test），仅使用 train 训练、test 评估。图像先做中心正方形裁剪，再缩放到 128×128。

## 环境

```powershell
python -m pip install -r requirements.txt
```

需要 PyTorch 和 torchvision。支持 CUDA 时会自动使用 GPU；也可通过 `--cpu` 强制使用 CPU。计算 FID 时，torchvision 首次下载 ImageNet 预训练 Inception-v3 权重。

数据目录应为：

```text
data/
  Eval/list_eval_partition.txt
  img_align_celeba/*.jpg
```

## 训练

```powershell
python train.py --epochs 50 --batch-size 32 --mask-strategy mixed
```

`mixed` 在随机块状遮挡和单个规则矩形遮挡之间随机选择；也可指定 `random` 或 `rectangle`。为快速检查流程，可使用 `--max-train-samples 256 --epochs 1 --batch-size 8 --workers 0`。训练完成后检查点保存在 `checkpoints/latest.pt`，每个 epoch 更新。

## 评估与对比

```powershell
python evaluate.py --max-samples 1000 --batch-size 32
```

评估会在同一 CelebA test 子集上分别使用随机遮挡和矩形遮挡，生成以下文件：

- `results/metrics.json`：两类 mask 的 FID、SSIM 和样本数。
- `results/random_comparison.png`、`results/rectangle_comparison.png`：原图、遮挡输入、修复结果和参考原图的对照图。

FID 基于 torchvision ImageNet 预训练 Inception-v3 的 2048 维池化特征，比较修复图与对应完整真实图像的特征分布；SSIM 在完整合成图（含被保留的可见像素）与原图之间计算。`--max-samples` 必须至少为 2，且指标受评估样本数量影响；正式对比建议增加样本数。主观评价请查看生成的对照图，重点观察五官结构、纹理连续性和整体自然度。

## 单张图片修复

```powershell
python inpaint.py --image data\img_align_celeba\000001.jpg --mask rectangle
```

输出默认写入 `results/inpainted.png`。单图推理当前提供随机块状 mask 或随机位置的矩形 mask。

## 测试

```powershell
python -m pytest
```
