# DenseDepth-pytorch
DenseDepth的pytorch实现
论文：High quility ....[url]

# 1.项目概述
单目视觉的深度估计，根据RGB图像，估计图像中每一个像素的深度。
# 2.数据集说明
- KITTI的深度数据集，具有深度估计的图像共计80K张，包含不同的场景
- groundtrue中有真实标注的像素点少，稀疏性大
- 使用colorlization[url]先对groundtrue中的系数值进行填充，再用于训练
# 网络介绍
# 代码使用说明
