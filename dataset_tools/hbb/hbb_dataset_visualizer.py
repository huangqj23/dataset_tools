import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

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
                 image_path: str,
                 label_path: str,
                 format: str,
                 save_path: Optional[str] = None,
                 show: bool = True,
                 thickness: int = 2) -> np.ndarray:
        """
        可视化单张图片的标注
        
        Args:
            image_path: 图片路径
            label_path: 标注文件路径
            format: 数据集格式，支持'voc'、'yolo'、'coco'
            save_path: 保存路径，如果为None则不保存
            show: 是否显示图片
            thickness: 边框线宽
            
        Returns:
            标注后的图片数组
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        height, width = image.shape[:2]
        
        # 根据不同格式读取标注
        if format.lower() == 'voc':
            boxes, labels = self._parse_voc(label_path)
        elif format.lower() == 'yolo':
            boxes, labels = self._parse_yolo(label_path, width, height)
        elif format.lower() == 'coco':
            boxes, labels = self._parse_coco(label_path, Path(image_path).name)
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
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, image)
            
        if show:
            cv2.imshow('image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return image
    
    def visualize_dataset(self,
                         dataset_dir: str,
                         format: str,
                         save_dir: Optional[str] = None) -> None:
        """
        批量可视化数据集
        
        Args:
            dataset_dir: 数据集根目录
            format: 数据集格式
            save_dir: 可视化结果保存目录
        """
        if format.lower() == 'voc':
            image_dir = os.path.join(dataset_dir, 'images')
            label_dir = os.path.join(dataset_dir, 'annotations')
            ext = '.xml'
        elif format.lower() == 'yolo':
            image_dir = os.path.join(dataset_dir, 'images')
            label_dir = os.path.join(dataset_dir, 'labels')
            ext = '.txt'
        elif format.lower() == 'coco':
            image_dir = os.path.join(dataset_dir, 'images')
            label_path = os.path.join(dataset_dir, 'annotations.json')
        else:
            raise ValueError(f"不支持的数据集格式: {format}")
            
        # 遍历图片
        for img_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_name)
            
            if format.lower() == 'coco':
                label_file = label_path
            else:
                base_name = os.path.splitext(img_name)[0]
                label_file = os.path.join(label_dir, base_name + ext)
            
            if save_dir:
                save_path = os.path.join(save_dir, img_name)
            else:
                save_path = None
                
            try:
                self.visualize(img_path, label_file, format, 
                             save_path=save_path, show=False)
            except Exception as e:
                print(f"处理 {img_name} 时出错: {str(e)}")
    
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
        save_path='output.jpg'
    )

    # 批量可视化数据集
    visualizer.visualize_dataset(
        dataset_dir='path/to/dataset',
        format='yolo',
        save_dir='path/to/output'
    )