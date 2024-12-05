import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

class OBBDatasetVisualizer:
    """旋转框数据集可视化工具，支持DOTA和COCO格式"""
    
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
                 thickness: int = 2,
                 font_scale: float = 0.5) -> np.ndarray:
        """
        可视化单张图片的标注
        
        Args:
            image_path: 图片路径
            label_path: 标注文件路径
            format: 数据集格式，支持'dota'、'coco'
            save_path: 保存路径，如果为None则不保存
            show: 是否显示图片
            thickness: 边框线宽
            font_scale: 字体大小
            
        Returns:
            标注后的图片数组
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 根据不同格式读取标注
        if format.lower() == 'dota':
            polygons, labels, difficulties = self._parse_dota(label_path)
        elif format.lower() == 'coco':
            polygons, labels = self._parse_coco(label_path, Path(image_path).name)
            difficulties = [0] * len(labels)  # COCO格式可能没有难度信息
        else:
            raise ValueError(f"不支持的数据集格式: {format}")
        
        # 绘制旋转框和标签
        for polygon, label, difficulty in zip(polygons, labels, difficulties):
            points = np.array(polygon).reshape((-1, 2)).astype(np.int32)
            color = self.colors.get(label, (0, 255, 0))
            
            # 绘制多边形
            cv2.polylines(image, [points], True, color, thickness)
            
            # 计算标签位置（使用多边形���左上角）
            label_text = f"{label}"
            if difficulty > 0:
                label_text += f" (diff:{difficulty})"
                
            x_min = min(points[:, 0])
            y_min = min(points[:, 1])
            
            # 绘制标签背景
            label_size, baseline = cv2.getTextSize(label_text, 
                                                 cv2.FONT_HERSHEY_SIMPLEX,
                                                 font_scale, 1)
            text_w, text_h = label_size
            cv2.rectangle(image, 
                         (x_min, y_min - text_h - baseline),
                         (x_min + text_w, y_min), 
                         color, -1)
            
            # 绘制标签文本
            cv2.putText(image, label_text,
                       (x_min, y_min - baseline),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, (255, 255, 255), 1)
        
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
                         save_dir: Optional[str] = None,
                         image_ext: str = '.png') -> None:
        """
        批量可视化数据集
        
        Args:
            dataset_dir: 数据集根目录
            format: 数据集格式
            save_dir: 可视化结果保存目录
            image_ext: 图片文件扩展名
        """
        if format.lower() == 'dota':
            image_dir = os.path.join(dataset_dir, 'images')
            label_dir = os.path.join(dataset_dir, 'labelTxt')
            
            # 遍历所有标注文件
            for txt_file in os.listdir(label_dir):
                if not txt_file.endswith('.txt'):
                    continue
                    
                base_name = txt_file[:-4]
                image_path = os.path.join(image_dir, base_name + image_ext)
                label_path = os.path.join(label_dir, txt_file)
                
                if save_dir:
                    save_path = os.path.join(save_dir, base_name + image_ext)
                else:
                    save_path = None
                    
                try:
                    self.visualize(image_path, label_path, format,
                                 save_path=save_path, show=False)
                except Exception as e:
                    print(f"处理 {base_name} 时出错: {str(e)}")
                    
        elif format.lower() == 'coco':
            image_dir = os.path.join(dataset_dir, 'images')
            label_path = os.path.join(dataset_dir, 'annotations.json')
            
            # 读取COCO标注文件
            with open(label_path, 'r') as f:
                coco_data = json.load(f)
            
            # 遍历所有图片
            for img_info in coco_data['images']:
                image_path = os.path.join(image_dir, img_info['file_name'])
                
                if save_dir:
                    save_path = os.path.join(save_dir, img_info['file_name'])
                else:
                    save_path = None
                    
                try:
                    self.visualize(image_path, label_path, format,
                                 save_path=save_path, show=False)
                except Exception as e:
                    print(f"处理 {img_info['file_name']} 时出错: {str(e)}")
        else:
            raise ValueError(f"不支持的数据集格式: {format}")
    
    def _parse_dota(self, txt_path: str) -> Tuple[List[List[float]], List[str], List[int]]:
        """解析DOTA格式的标注文件"""
        polygons = []
        labels = []
        difficulties = []
        
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
                difficulty = int(parts[9]) if len(parts) > 9 else 0
                
                polygons.append(points)
                labels.append(category)
                difficulties.append(difficulty)
                
        return polygons, labels, difficulties
    
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
    # 创建可视化器实例
    visualizer = OBBDatasetVisualizer()

    # 设置类别名称
    categories = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 
                'tennis-court', 'basketball-court', 'ground-track-field', 
                'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 
                'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']
    visualizer.set_class_names(categories)

    # 可视化单张DOTA格式图片
    visualizer.visualize(
        image_path='path/to/image.png',
        label_path='path/to/label.txt',
        format='dota',
        save_path='output.png'
    )

    # 批量可视化COCO格式数据集
    visualizer.visualize_dataset(
        dataset_dir='path/to/dataset',
        format='coco',
        save_dir='path/to/output'
    )