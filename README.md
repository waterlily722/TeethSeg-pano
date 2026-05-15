# ECE3070 课程项目：牙齿全景 X 光片实例分割

本仓库用于 ECE3070 课程项目，核心实验脚本与报告对齐如下：

- 课程报告文档：ECE3070 Group Coursework - V2.pdf
- 主要实验 Notebook：teeth_three_methods.ipynb

项目目标是在全景牙片中完成 32 个 FDI 牙位的实例分割，并比较三类方法：

1. Active Contour（传统方法，无监督训练）
2. U-Net（仅使用 X 光图像）
3. OralBBNet（X 光图像 + YOLO 边界框先验）

## 与课程项目对应关系

本仓库中的实验流程与课程项目写作结构一一对应：

1. 数据索引与预处理
2. 训练/验证/测试划分
3. 三种方法训练与推理
4. 阈值搜索与定量评估
5. 定性可视化与实验结论

在 Notebook 中，U-Net 与 OralBBNet 都会在验证集上自动搜索最佳二值化阈值，再用于测试集评估，指标包含：

- Binary Dice
- Binary IoU
- Mean Channel Dice
- 分组 Dice（门牙/尖牙/前磨牙/磨牙）

## 仓库说明

- teeth_three_methods.ipynb 是课程实验主入口。
- results/ 存放指标、训练曲线与可视化结果。
- cache/ 存放预处理缓存。

## 环境与依赖

建议使用 Python 3.10+，并在支持 TensorFlow GPU 的环境中运行。

安装依赖示例：

```bash
pip install numpy pandas matplotlib pillow tifffile scikit-image scikit-learn opencv-python tensorflow keras
```

## 运行方式

1. 进入仓库根目录后打开 Notebook：

- teeth_three_methods.ipynb

2. 按单元顺序运行，关键步骤如下：

- 配置路径、训练参数、随机种子
- 加载并缓存预处理数据
- 训练 U-Net 与 OralBBNet
- 在验证集搜索最佳阈值
- 在测试集输出指标与可视化

3. 主要输出文件：

- results/metrics.csv
- results/unet_history.csv
- results/oralbbnet_history.csv
- results/qualitative_examples.png
- results/checkpoints/*.weights.h5

## 版本管理说明

为避免提交大文件，仓库已配置忽略以下内容：

- cache/
- results/checkpoints/
- *.weights.h5
- *.pth
- *.pt
- *.ckpt

这与课程项目提交流程一致：保留代码、报告、指标与可视化结果，排除缓存与模型权重。

## 数据与参考

- UFBA-425 数据集：https://figshare.com/articles/dataset/UFBA-425/29827475
- OralBBNet 论文：https://arxiv.org/abs/2406.03747

Hyperparameter Setup: The optimal results were obtained with the Adam optimizer with a learning rate of 0.0003 with a momentum of 0.99, a batch size of 2 and dropout rate of 0.12 and regularization constant λ of 0.1 and training over 60 epochs. The learning rate was halved if validation loss did not improve over a period of 5 epochs. The inference settings for YOLOv8 include a confidence threshold of 0.5 and an Intersection over Union (IoU) threshold of 0.5. 