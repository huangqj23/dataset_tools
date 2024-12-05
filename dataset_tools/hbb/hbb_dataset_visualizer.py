import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm

class HBBDatasetVisualizer:
    """数据集可视化工具，支持VOC、YOLO和COCO格式"""
    
    def __init__(self):
        self.colors = None
        self.class_names = []
        
    def set_class_names(self, class_names: List[str]):
        """设置类别名称列表"""
        self.class_names = class_names
        # 为每个类别随机生成一个颜色
        np.random.seed(42)
        self.colors = {name: tuple(map(int, np.random.randint(0, 255, 3))) 
                      for name in class_names}
    
    def visualize(self,
                 image_path: Union[str, Path],
                 label_path: Union[str, Path],
                 format: str,
                 save_dir: Optional[Union[str, Path]] = None,
                 show: bool = True,
                 thickness: int = 2,
                 num_workers: int = 4) -> Optional[np.ndarray]:
        """
        可视化数据集标注。支持单张图片或整个数据集的可视化。
        
        Args:
            image_path: 图片路径或图片目录
            label_path: 标注文件路径或标注目录(COCO格式为json文件路径，其他格式为标注目录)
            format: 数据集格式，支持'voc'、'yolo'、'coco'
            save_dir: 保存目录，如果为None则不保存
            show: 是否显示图片
            thickness: 边框线宽
            num_workers: 并行处理的线程数
            
        Returns:
            如果是单张图片，返回标注后的图片数组；如果是数据集，返回None
        """
        image_path = Path(image_path)
        label_path = Path(label_path)
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # 判断是单张图片还是数据集
        if image_path.is_file():
            return self._visualize_single(
                image_path, label_path, format, 
                save_dir / image_path.name if save_dir else None,
                show, thickness
            )
        
        # 处理数据集目录
        if not image_path.is_dir():
            raise ValueError(f"图片路径不存在或无效: {image_path}")
        
        # 验证标注路径
        if format.lower() == 'coco':
            if not label_path.is_file():
                raise ValueError(f"COCO格式需要提供标注文件: {label_path}")
        else:
            if not label_path.is_dir():
                raise ValueError(f"标注目录不存在: {label_path}")
        
        # 收集所有需要处理的图片
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
            print("警告: 未找到任何图片文件")
            return None
        
        # 创建处理任务
        def process_image(img_file: Path) -> Tuple[bool, str]:
            try:
                if format.lower() == 'coco':
                    ann_file = label_path  # COCO格式使用同一个标注文件
                else:
                    ann_file = label_path / f"{img_file.stem}{self._get_label_ext(format)}"
                    if not ann_file.exists():
                        return False, f"未找到对应的标注文件 {ann_file}"
                        
                self._visualize_single(
                    img_file, 
                    ann_file,
                    format,
                    save_dir / img_file.name if save_dir else None,
                    show=False,  # 数据集模式下不显示
                    thickness=thickness
                )
                return True, ""
            except Exception as e:
                return False, f"处理出错: {str(e)}"
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 创建进度条
            results = list(tqdm(
                executor.map(process_image, image_files),
                total=len(image_files),
                desc="Processing images",
                unit="img"
            ))
        
        # 统计处理结果
        success_count = sum(1 for success, _ in results if success)
        print(f"\n处理完成: 成功 {success_count}/{len(image_files)} 张图片")
        
        # 显示错误信息
        for success, message in results:
            if not success and message:
                print(f"错误: {message}")
                
        return None
    
    def _visualize_single(self,
                         image_path: Path,
                         label_path: Path,
                         format: str,
                         save_path: Optional[Path] = None,
                         show: bool = True,
                         thickness: int = 2) -> np.ndarray:
        """处理单张图片的可视化"""
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        height, width = image.shape[:2]
        
        # 根据不同格式读取标注
        if format.lower() == 'voc':
            boxes, labels = self._parse_voc(label_path)
        elif format.lower() == 'yolo':
            boxes, labels = self._parse_yolo(label_path, width, height)
        elif format.lower() == 'coco':
            boxes, labels = self._parse_coco(label_path, image_path.name)
        else:
            raise ValueError(f"不支持的数据集格式: {format}")
        
        # 绘制边框和标签
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            color = self.colors.get(label, (0, 255, 0))
            
            # 绘制边框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制标签
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_w, text_h = label_size
            cv2.rectangle(image, (x1, y1 - text_h - baseline), 
                         (x1 + text_w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - baseline), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), image)
            
        if show:
            cv2.imshow('image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return image
    
    def _get_label_ext(self, format: str) -> str:
        """获取不同格式的标注文件扩展名"""
        if format.lower() == 'voc':
            return '.xml'
        elif format.lower() == 'yolo':
            return '.txt'
        elif format.lower() == 'coco':
            return '.json'
        else:
            raise ValueError(f"不支持的数据集格式: {format}")
    
    def _parse_voc(self, xml_path: str) -> Tuple[List[List[float]], List[str]]:
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
    
    def _parse_yolo(self, txt_path: str, img_width: int, img_height: int) -> Tuple[List[List[float]], List[str]]:
        """解析YOLO格式的标注文件"""
        boxes = []
        labels = []
        
        with open(txt_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = line.strip().split()
                    class_id = int(data[0])
                    if class_id >= len(self.class_names):
                        raise ValueError(f"类别ID {class_id} 超出类别列表范围")
                        
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
                    labels.append(self.class_names[class_id])
        
        return boxes, labels
    
    def _parse_coco(self, json_path: str, image_name: str) -> Tuple[List[List[float]], List[str]]:
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
            raise ValueError(f"在COCO标注中未找到图片: {image_name}")
            
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
    # 创建可视化器实例
    visualizer = HBBDatasetVisualizer()

    # 设置类别名称（对YOLO格式必需，其他格式可选）
    class_names = ['person', 'car', 'dog']  # 按类别ID顺序排列
    visualizer.set_class_names(class_names)

    # 可视化单张图片
    visualizer.visualize(
        image_path='path/to/image.jpg',
        label_path='path/to/label.xml',
        format='voc',
        save_dir='path/to/output'
    )