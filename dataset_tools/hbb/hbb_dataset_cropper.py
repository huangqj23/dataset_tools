import cv2
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np

class HBBDatasetCropper:
    """水平框数据集剪切工具，支持将标注框剪切到对应类别文件夹"""
    
    def __init__(self):
        self.supported_formats = ['yolo', 'coco', 'voc']
        
    def crop(self,
            image_path: Union[str, Path],
            label_path: Union[str, Path],
            output_dir: Union[str, Path],
            format: str,
            classes_file: Optional[Union[str, Path]] = None,
            num_workers: int = 4):
        """
        剪切数据集中的标注框到对应类别文件夹
        
        Args:
            image_path: 图片路径或图片目录
            label_path: 标注文件路径或标注目录
            output_dir: 输出根目录，每个类别会创建独立子文件夹
            format: 数据集格式，支持'yolo'、'coco'、'voc'
            classes_file: YOLO格式的类别文件路径
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
                
        if format == 'yolo' and (not classes_file or not Path(classes_file).is_file()):
            raise ValueError("YOLO格式需要提供classes.txt文件")
            
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
            
        # 读取YOLO类别文件
        if format == 'yolo':
            with open(classes_file, 'r') as f:
                classes = f.read().strip().split('\n')
        
        # 创建处理任务
        def process_image(img_file: Path) -> Tuple[bool, str]:
            try:
                # 读取图片
                img = cv2.imread(str(img_file))
                if img is None:
                    return False, f"无法读取图片: {img_file}"
                
                # 根据格式读取标注
                if format == 'voc':
                    boxes, labels = self._parse_voc(label_path / f"{img_file.stem}.xml")
                elif format == 'yolo':
                    boxes, labels = self._parse_yolo(
                        label_path / f"{img_file.stem}.txt",
                        img.shape[1], img.shape[0],
                        classes
                    )
                else:  # coco
                    boxes, labels = self._parse_coco(label_path, img_file.name)
                
                # 剪切每个标注框
                for box, label in zip(boxes, labels):
                    x1, y1, x2, y2 = map(int, box)
                    # 确保坐标在图片范围内
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img.shape[1], x2)
                    y2 = min(img.shape[0], y2)
                    
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
    
    def _parse_voc(self, xml_path: Path) -> Tuple[List[List[float]], List[str]]:
        """解析VOC格式的XML文件"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(name)
            
        return boxes, labels
    
    def _parse_yolo(self, txt_path: Path, img_width: int, img_height: int, 
                   classes: List[str]) -> Tuple[List[List[float]], List[str]]:
        """解析YOLO格式的标注文件"""
        boxes = []
        labels = []
        
        with open(txt_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = line.strip().split()
                    class_id = int(data[0])
                    if class_id >= len(classes):
                        continue
                        
                    # 转换YOLO格式到绝对坐标
                    x_center, y_center, w, h = map(float, data[1:5])
                    x_center *= img_width
                    y_center *= img_height
                    w *= img_width
                    h *= img_height
                    
                    xmin = x_center - w/2
                    ymin = y_center - h/2
                    xmax = x_center + w/2
                    ymax = y_center + h/2
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(classes[class_id])
        
        return boxes, labels
    
    def _parse_coco(self, json_path: Path, image_name: str) -> Tuple[List[List[float]], List[str]]:
        """解析COCO格式的标注文件"""
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
            
        # 创建类别ID到名称的映射
        cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # 找到对应图片的ID
        image_id = None
        for img in coco_data['images']:
            if img['file_name'] == image_name:
                image_id = img['id']
                break
                
        if image_id is None:
            return [], []
            
        boxes = []
        labels = []
        
        # 获取该图片的所有标注
        for ann in coco_data['annotations']:
            if ann['image_id'] == image_id:
                bbox = ann['bbox']  # [x,y,width,height]
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[0] + bbox[2]
                ymax = bbox[1] + bbox[3]
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cat_id_to_name[ann['category_id']])
                
        return boxes, labels

if __name__ == '__main__':
    # 使用示例
    cropper = HBBDatasetCropper()
    
    # YOLO格式
    cropper.crop(
        image_path='path/to/images',
        label_path='path/to/labels',
        output_dir='path/to/output',
        format='yolo',
        classes_file='path/to/classes.txt'
    )
    
    # VOC格式
    cropper.crop(
        image_path='path/to/images',
        label_path='path/to/annotations',
        output_dir='path/to/output',
        format='voc'
    )
    
    # COCO格式
    cropper.crop(
        image_path='path/to/images',
        label_path='path/to/annotations.json',
        output_dir='path/to/output',
        format='coco'
    ) 