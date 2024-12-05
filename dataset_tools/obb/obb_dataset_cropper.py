import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class OBBDatasetCropper:
    """旋转框数据集剪切工具，支持将标注框剪切到对应类别文件夹"""
    
    def __init__(self):
        self.supported_formats = ['dota', 'coco']
        
    def crop(self,
            image_path: Union[str, Path],
            label_path: Union[str, Path],
            output_dir: Union[str, Path],
            format: str,
            num_workers: int = 4):
        """
        剪切数据集中的标注框到对应类别文件夹。
        对于旋转框，会转换为其最小外接矩形进行剪切。
        
        Args:
            image_path: 图片路径或图片目录
            label_path: 标注文件路径或标注目录
            output_dir: 输出根目录，每个类别会创建独立子文件夹
            format: 数据集格式，支持'dota'、'coco'
            num_workers: 并行处理的线程数
        """
        image_path = Path(image_path)
        label_path = Path(label_path)
        output_dir = Path(output_dir)
        
        if format not in self.supported_formats:
            raise ValueError(f"不支持的格式。支持的格式为: {self.supported_formats}")
            
        # 验证路径
        if not image_path.is_dir():
            raise ValueError(f"图片目录不存在: {image_path}")
            
        if format == 'coco':
            if not label_path.is_file():
                raise ValueError(f"COCO标注文件不存在: {label_path}")
        else:
            if not label_path.is_dir():
                raise ValueError(f"标注目录不存在: {label_path}")
        
        # 收集所有图片文件
        image_patterns = [
            "*.[jJ][pP][gG]",      # jpg, JPG
            "*.[jJ][pP][eE][gG]",  # jpeg, JPEG
            "*.[pP][nN][gG]",      # png, PNG
            "*.[tT][iI][fF]",      # tif, TIF
            "*.[tT][iI][fF][fF]"   # tiff, TIFF
        ]
        
        image_files = []
        for pattern in image_patterns:
            image_files.extend(list(image_path.glob(pattern)))
            
        if not image_files:
            raise ValueError(f"未找到任何图片文件: {image_path}")
        
        # 创建处理任务
        def process_image(img_file: Path) -> Tuple[bool, str]:
            try:
                # 读取图片
                img = cv2.imread(str(img_file))
                if img is None:
                    return False, f"无法读取图片: {img_file}"
                
                # 根据格式读取标注
                if format == 'dota':
                    polygons, labels = self._parse_dota(label_path / f"{img_file.stem}.txt")
                else:  # coco
                    polygons, labels = self._parse_coco(label_path, img_file.name)
                
                # 剪切每个标注框
                for polygon, label in zip(polygons, labels):
                    # 将多边形点转换为numpy数组
                    points = np.array(polygon).reshape((-1, 2))
                    
                    # 计算最小外接矩形
                    rect = cv2.minAreaRect(points.astype(np.float32))
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    # 获取最小外接矩形的边界
                    x_coords = box[:, 0]
                    y_coords = box[:, 1]
                    x1, x2 = max(0, min(x_coords)), min(img.shape[1], max(x_coords))
                    y1, y2 = max(0, min(y_coords)), min(img.shape[0], max(y_coords))
                    
                    # 确保有效的裁剪区域
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # 剪切图片
                    cropped = img[y1:y2, x1:x2]
                    if cropped.size == 0:
                        continue
                    
                    # 创建类别目录
                    save_dir = output_dir / label
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 保存剪切图片，文件名格式：原图名_x1_y1_x2_y2.jpg
                    save_path = save_dir / f"{img_file.stem}_{x1}_{y1}_{x2}_{y2}{img_file.suffix}"
                    cv2.imwrite(str(save_path), cropped)
                
                return True, ""
            except Exception as e:
                return False, f"处理出错: {str(e)}"
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_image, image_files),
                total=len(image_files),
                desc="Cropping images",
                unit="img"
            ))
        
        # 统计处理结果
        success_count = sum(1 for success, _ in results if success)
        print(f"\n处理完成: 成功 {success_count}/{len(image_files)} 张图片")
        
        # 显示错误信息
        for success, message in results:
            if not success and message:
                print(f"错误: {message}")
    
    def _parse_dota(self, txt_path: Path) -> Tuple[List[List[float]], List[str]]:
        """解析DOTA格式的标注文件"""
        polygons = []
        labels = []
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            # 跳过第一行
            next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                    
                # 解析坐标点和类别
                points = list(map(float, parts[:8]))
                category = parts[8]
                
                polygons.append(points)
                labels.append(category)
                
        return polygons, labels
    
    def _parse_coco(self, json_path: Path, image_name: str) -> Tuple[List[List[float]], List[str]]:
        """解析COCO格式的标注文件"""
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
            
        # 创建类别ID到名称的映射
        cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # ��到对应图片的ID
        image_id = None
        for img in coco_data['images']:
            if img['file_name'] == image_name:
                image_id = img['id']
                break
                
        if image_id is None:
            return [], []
            
        polygons = []
        labels = []
        
        # 获取该图片的所有标注
        for ann in coco_data['annotations']:
            if ann['image_id'] == image_id:
                if 'segmentation' in ann and ann['segmentation']:
                    points = ann['segmentation'][0]  # 假设是多边形格式
                    if len(points) == 8:  # 确保是4个点的多边形
                        polygons.append(points)
                        labels.append(cat_id_to_name[ann['category_id']])
                
        return polygons, labels

if __name__ == '__main__':
    # 使用示例
    cropper = OBBDatasetCropper()
    
    # DOTA格式
    cropper.crop(
        image_path='path/to/images',
        label_path='path/to/labelTxt',
        output_dir='path/to/output',
        format='dota'
    )
    
    # COCO格式
    cropper.crop(
        image_path='path/to/images',
        label_path='path/to/annotations.json',
        output_dir='path/to/output',
        format='coco'
    ) 