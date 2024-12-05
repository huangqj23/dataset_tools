# Dataset Format Converter

一个功能强大的目标检测数据集格式转换工具，支持常见数据集格式之间的互相转换和可视化。

## 功能特点

### 水平矩形框数据集转换
支持以下格式之间的互相转换：
- VOC (.xml)
- YOLO (.txt)
- COCO (.json)

### 旋转框数据集转换
支持以下格式之间的转换：
- DOTA (.txt)
- COCO (.json)

### 可视化功能
- 支持水平框和旋转框的可视化
- 支持批量可视化
- 可自定义可视化参数（框的颜色、线宽等）

## 数据集格式要求

### VOC格式

## 依赖库
- opencv-python
- numpy
- tqdm

## 安装

### 方法1：通过pip安装（推荐）
```
pip install dataset_tools
```

### 方法2：从源码安装
```bash
# 克隆仓库
git clone https://github.com/yourusername/dataset_tools.git
cd dataset_tools

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

### 方法3：仅安装依赖
如果你不想安装包，只想使用代码，可以只安装依赖：
```bash
pip install -r requirements.txt
```
