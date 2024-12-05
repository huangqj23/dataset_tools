import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

class OBBDatasetConverter:
    """旋转框数据集格式转换器，支持DOTA和COCO格式之间的转换"""
    
    def __init__(self):
        self.categories = []
        self.category_to_id = {}
    
    def set_categories(self, categories: List[str]):
        """设置类别列表"""
        self.categories = categories
        self.category_to_id = {cat: idx + 1 for idx, cat in enumerate(categories)}
    
    def dota_to_coco(self,
                     dota_dir: str,
                     output_path: str,
                     image_ext: str = '.png') -> None:
        """
        将DOTA格式数据集转换为COCO格式
        
        Args:
            dota_dir: DOTA数据集根目录，包含images和labelTxt子目录
            output_path: COCO格式标注文件的保存路径
            image_ext: 图片文件扩展名
        """
        if not self.categories:
            raise ValueError("请先使用set_categories设置类别列表")
            
        # 准备COCO格式数据结构
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": idx + 1, "name": cat} 
                for idx, cat in enumerate(self.categories)
            ]
        }
        
        image_dir = os.path.join(dota_dir, 'images')
        label_dir = os.path.join(dota_dir, 'labelTxt')
        
        # 确保目录存在
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            raise ValueError(f"数据集目录结构不正确: {dota_dir}")
            
        ann_id = 1
        
        # 遍历所有标注文件
        for txt_file in tqdm(os.listdir(label_dir), desc="Converting DOTA to COCO"):
            if not txt_file.endswith('.txt'):
                continue
                
            image_id = len(coco_data["images"]) + 1
            base_name = txt_file[:-4]
            image_file = base_name + image_ext
            image_path = os.path.join(image_dir, image_file)
            
            # 读取图片尺寸
            try:
                import cv2
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Warning: Cannot read image {image_path}")
                    continue
                height, width = img.shape[:2]
            except Exception as e:
                print(f"Error reading image {image_path}: {str(e)}")
                continue
            
            # 添加图片信息
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_file,
                "height": height,
                "width": width
            })
            
            # 解析标注文件
            try:
                with open(os.path.join(label_dir, txt_file), 'r', encoding='utf-8') as f:
                    # 跳过第一行（可能包含标题）
                    next(f)
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 9:
                            continue
                            
                        # 解析坐标点和类别
                        points = list(map(float, parts[:8]))
                        category = parts[8]
                        difficulty = int(parts[9]) if len(parts) > 9 else 0
                        
                        # 检查类别是否在预定义列表中
                        if category not in self.category_to_id:
                            continue
                            
                        # 转换为COCO格式
                        x_coords = points[0::2]
                        y_coords = points[1::2]
                        
                        # 计算边界框
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        w = x_max - x_min
                        h = y_max - y_min
                        
                        # 添加标注
                        coco_data["annotations"].append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": self.category_to_id[category],
                            "segmentation": [points],  # 多边形坐标
                            "bbox": [x_min, y_min, w, h],  # 水平外接矩形
                            "bbox_mode": "poly",  # 表示这是一个多边形标注
                            "area": self._polygon_area(points),
                            "iscrowd": 0,
                            "difficulty": difficulty
                        })
                        ann_id += 1
                        
            except Exception as e:
                print(f"Error processing {txt_file}: {str(e)}")
                continue
        
        # 保存COCO格式数据
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
    
    def coco_to_dota(self,
                     coco_path: str,
                     image_dir: str,
                     output_dir: str) -> None:
        """
        将COCO格式数据集转换为DOTA格式
        
        Args:
            coco_path: COCO格式标注文件路径
            image_dir: 图片目录路径
            output_dir: 输出目录路径
        """
        # 读取COCO数据
        with open(coco_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
            
        # 创建ID到类别名称的映射
        id_to_category = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # 创建图片ID到文件名的映射
        image_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # 按图片ID组织标注
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # 创建输出目录
        output_image_dir = os.path.join(output_dir, 'images')
        output_label_dir = os.path.join(output_dir, 'labelTxt')
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        
        # 处理每张图片
        for image_id, annotations in tqdm(annotations_by_image.items(), desc="Converting COCO to DOTA"):
            image_name = image_id_to_name[image_id]
            base_name = os.path.splitext(image_name)[0]
            
            # 写入标注文件
            with open(os.path.join(output_label_dir, f"{base_name}.txt"), 'w', encoding='utf-8') as f:
                # 写入标题行
                f.write('imagesource:GoogleEarth\n')
                # 写入标注
                for ann in annotations:
                    if 'segmentation' not in ann or not ann['segmentation']:
                        continue
                        
                    category = id_to_category[ann['category_id']]
                    points = ann['segmentation'][0]  # 假设是多边形格式
                    difficulty = ann.get('difficulty', 0)
                    
                    # 确保点的数量正确（4个点，8个坐标）
                    if len(points) != 8:
                        continue
                        
                    # 写入DOTA格式：x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
                    coords = ' '.join(map(str, points))
                    f.write(f"{coords} {category} {difficulty}\n")
            
            # 复制图片文件
            src_image = os.path.join(image_dir, image_name)
            dst_image = os.path.join(output_image_dir, image_name)
            if os.path.exists(src_image):
                import shutil
                shutil.copy2(src_image, dst_image)
    
    def _polygon_area(self, points: List[float]) -> float:
        """计算多边形面积"""
        x = points[0::2]
        y = points[1::2]
        return 0.5 * abs(sum(i * j for i, j in zip(x, y[1:] + y[:1])) - 
                        sum(i * j for i, j in zip(x[1:] + x[:1], y))) 
        
if __name__ == '__main__':
    # 创建转换器实例
    converter = OBBDatasetConverter()

    # 设置类别列表
    categories = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 
                'tennis-court', 'basketball-court', 'ground-track-field', 
                'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 
                'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']
    converter.set_categories(categories)

    # DOTA转COCO
    converter.dota_to_coco(
        dota_dir='path/to/dota/dataset',
        output_path='path/to/output/annotations.json',
        image_ext='.png'
    )

    # COCO转DOTA
    converter.coco_to_dota(
        coco_path='path/to/coco/annotations.json',
        image_dir='path/to/coco/images',
        output_dir='path/to/output/dota'
    )