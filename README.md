# DenseDepth-pytorch
DenseDepth的pytorch实现  
论文：[High Quality Monocular Depth Estimation via Transfer Learning](https://arxiv.org/abs/1812.11941)  
原始代码：[Keras实现](https://github.com/ialhashim/DenseDepth)

# 1.项目概述
单目视觉的深度估计，根据RGB图像，估计图像中每一个像素的深度。
# 2.数据集说明
- KITTI的深度数据集，具有深度估计的图像共计80K张，包含不同的场景
- groundtrue中有真实标注的像素点少，稀疏性大
- 先根据RGB图像的特征，对groundtrue中的缺失值进行填充，再用于训练
# 3.网络介绍
使用预训练的DenseNet-169作为Encoder，使用了四个直连结构以补充细节，Decoder阶段级联upsample和Decoder的特征图，再进行卷积，经过四次的Upsample和卷积，输出预测值
# 4.代码使用说明
- 首先使用fill_death_map.py,填充groundtrue的缺失值，并将图像的路径汇总，生成assets中的csv文件；
- 执行train_depth.py即可直接开始训练，并保存验证集上分数最高的模型，生成的日志和模型在runs中
