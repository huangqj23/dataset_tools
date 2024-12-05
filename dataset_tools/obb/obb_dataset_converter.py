import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor

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
                     dota_dir: Union[str, Path],
                     output_path: Union[str, Path],
                     image_ext: str = '.png',
                     num_workers: int = 4) -> None:
        """
        将DOTA格式数据集转换为COCO格式
        
        Args:
            dota_dir: DOTA数据集图片目录
            output_path: COCO格式标注文件的保存路径
            image_ext: 图片文件扩展名
            num_workers: 并行处理的线程数
        """
        dota_dir = Path(dota_dir)
        output_path = Path(output_path)
        
        if not self.categories:
            raise ValueError("请先使用set_categories设置类别列表")
            
        # 验证路径
        if not dota_dir.is_dir():
            raise ValueError(f"图片目录不存在: {dota_dir}")
            
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备COCO格式数据结构
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": idx + 1, "name": cat} 
                for idx, cat in enumerate(self.categories)
            ]
        }
        
        image_dir = dota_dir / 'images'
        label_dir = dota_dir / 'labelTxt'
        
        # 确保目录存在
        if not image_dir.exists() or not label_dir.exists():
            raise ValueError(f"数据集目录结构不正确: {dota_dir}")
            
        def process_txt(txt_file: Path, image_id: int) -> Tuple[Dict, List[Dict]]:
            try:
                image_file = txt_file.stem + image_ext
                image_path = image_dir / image_file
                
                # 读取图片尺寸
                img = cv2.imread(str(image_path))
                if img is None:
                    return None, []
                height, width = img.shape[:2]
                
                # 准备图片信息
                image_info = {
                    "id": image_id,
                    "file_name": image_file,
                    "height": height,
                    "width": width
                }
                
                # 读取标注信息
                annotations = []
                with open(txt_file, 'r', encoding='utf-8') as f:
                    next(f)  # 跳过第一行
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 9:
                            continue
                            
                        points = list(map(float, parts[:8]))
                        category = parts[8]
                        difficulty = int(parts[9]) if len(parts) > 9 else 0
                        
                        if category not in self.category_to_id:
                            continue
                            
                        x_coords = points[0::2]
                        y_coords = points[1::2]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        w = x_max - x_min
                        h = y_max - y_min
                        
                        annotations.append({
                            "image_id": image_id,
                            "category_id": self.category_to_id[category],
                            "segmentation": [points],
                            "bbox": [x_min, y_min, w, h],
                            "bbox_mode": "poly",
                            "area": self._polygon_area(points),
                            "iscrowd": 0,
                            "difficulty": difficulty
                        })
                        
                return image_info, annotations
            except Exception as e:
                print(f"处理 {txt_file} 时出错: {str(e)}")
                return None, []
        
        # 使用线程池并行处理
        txt_files = list(label_dir.glob('*.txt'))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for image_id, txt_file in enumerate(txt_files, start=1):
                future = executor.submit(process_txt, txt_file, image_id)
                futures.append(future)
            
            # 收集结果
            ann_id = 1
            for future in tqdm(futures, desc="Converting DOTA to COCO"):
                image_info, annotations = future.result()
                if image_info:
                    coco_data["images"].append(image_info)
                    for ann in annotations:
                        ann["id"] = ann_id
                        coco_data["annotations"].append(ann)
                        ann_id += 1
        
        # 保存COCO格式数据
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
    
    def coco_to_dota(self,
                     coco_path: Union[str, Path],
                     image_dir: Union[str, Path],
                     output_dir: Union[str, Path]) -> None:
        """
        将COCO格式数据集转换为DOTA格式
        
        Args:
            coco_path: COCO格式标注文件路径
            image_dir: 图片目录路径
            output_dir: 输出目录路径
        """
        coco_path = Path(coco_path)
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        
        # 验证路径
        if not coco_path.is_file():
            raise ValueError(f"COCO标注文件不存在: {coco_path}")
        if not image_dir.is_dir():
            raise ValueError(f"图片目录不存在: {image_dir}")
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # 处理每张图片
        for image_id, annotations in tqdm(annotations_by_image.items(), desc="Converting COCO to DOTA"):
            image_name = image_id_to_name[image_id]
            base_name = os.path.splitext(image_name)[0]
            
            # 写入标注文件
            with open(os.path.join(output_dir, 'labelTxt', f"{base_name}.txt"), 'w', encoding='utf-8') as f:
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
            dst_image = os.path.join(output_dir, 'images', image_name)
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