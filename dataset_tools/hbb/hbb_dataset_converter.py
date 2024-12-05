import os
import json
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm
from glob import glob
from typing import List, Dict, Tuple

class HBBDatasetConverter:
    def __init__(self):
        self.supported_formats = ['yolo', 'coco', 'voc']
        
    def convert(self, source_dir: str, source_format: str, 
                target_format: str, output_path: str,
                classes_file: str = None):
        """
        转换数据集格式
        
        Args:
            source_dir: 源数据集目录
            source_format: 源格式 ('yolo', 'coco', 'voc')
            target_format: 目标格式 ('yolo', 'coco', 'voc')
            output_path: 输出路径
            classes_file: 类别文件路径（YOLO格式需要）
        """
        if source_format not in self.supported_formats or target_format not in self.supported_formats:
            raise ValueError(f"不支持的格式。支持的格式为: {self.supported_formats}")
            
        # 读取数据集
        if source_format == 'yolo':
            dataset = self._read_yolo(source_dir, classes_file)
        elif source_format == 'coco':
            dataset = self._read_coco(source_dir)
        else:  # VOC
            dataset = self._read_voc(source_dir)
            
        # 转换并保存
        if target_format == 'yolo':
            self._save_yolo(dataset, output_path)
        elif target_format == 'coco':
            self._save_coco(dataset, output_path)
        else:  # VOC
            self._save_voc(dataset, output_path)

    def _read_yolo(self, yolo_dir: str, classes_file: str) -> Dict:
        """读取YOLO格式数据集"""
        dataset = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # 读取类别
        with open(classes_file, 'r') as f:
            classes = f.read().strip().split('\n')
            for i, cls in enumerate(classes):
                dataset['categories'].append({
                    'id': i,
                    'name': cls,
                    'supercategory': 'none'
                })
        
        # 读取图片和标注
        ann_id = 0
        img_files = glob(os.path.join(yolo_dir, "*.jpg")) + glob(os.path.join(yolo_dir, "*.png"))
        
        for img_id, img_path in enumerate(tqdm(img_files)):
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            
            dataset['images'].append({
                'id': img_id,
                'file_name': os.path.basename(img_path),
                'width': width,
                'height': height,
                'path': img_path
            })
            
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            if not os.path.exists(txt_path):
                continue
                
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    cls_id, x_center, y_center, w, h = map(float, line.strip().split())
                    
                    x = (x_center - w/2) * width
                    y = (y_center - h/2) * height
                    w = w * width
                    h = h * height
                    
                    dataset['annotations'].append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': int(cls_id),
                        'bbox': [x, y, w, h],
                        'area': w * h,
                        'iscrowd': 0
                    })
                    ann_id += 1
                    
        return dataset

    def _read_coco(self, coco_path: str) -> Dict:
        """读取COCO格式数据集"""
        with open(coco_path, 'r') as f:
            return json.load(f)

    def _read_voc(self, voc_dir: str) -> Dict:
        """读取VOC格式数据集"""
        dataset = {
            'images': [],
            'annotations': [],
            'categories': set()
        }
        
        ann_id = 0
        for img_id, xml_file in enumerate(tqdm(glob(os.path.join(voc_dir, "*.xml")))):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 读取图片信息
            filename = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            dataset['images'].append({
                'id': img_id,
                'file_name': filename,
                'width': width,
                'height': height,
                'path': os.path.join(voc_dir, filename)
            })
            
            # 读取标注信息
            for obj in root.findall('object'):
                name = obj.find('name').text
                dataset['categories'].add(name)
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                w = xmax - xmin
                h = ymax - ymin
                
                dataset['annotations'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_name': name,
                    'bbox': [xmin, ymin, w, h],
                    'area': w * h,
                    'iscrowd': 0
                })
                ann_id += 1
        
        # 转换categories为列表格式
        categories = []
        for i, name in enumerate(sorted(dataset['categories'])):
            categories.append({
                'id': i,
                'name': name,
                'supercategory': 'none'
            })
        dataset['categories'] = categories
        
        # 更新annotation中的category_id
        name_to_id = {cat['name']: cat['id'] for cat in categories}
        for ann in dataset['annotations']:
            ann['category_id'] = name_to_id[ann['category_name']]
            del ann['category_name']
            
        return dataset

    def _save_yolo(self, dataset: Dict, output_dir: str):
        """保存为YOLO格式"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存类别文件
        with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
            for cat in dataset['categories']:
                f.write(f"{cat['name']}\n")
        
        # 保存标注文件
        for img in dataset['images']:
            img_id = img['id']
            width = img['width']
            height = img['height']
            
            # 获取该图片的所有标注
            annotations = [ann for ann in dataset['annotations'] if ann['image_id'] == img_id]
            
            # 创建txt文件
            txt_path = os.path.join(output_dir, os.path.splitext(img['file_name'])[0] + '.txt')
            with open(txt_path, 'w') as f:
                for ann in annotations:
                    x, y, w, h = ann['bbox']
                    # 转换为YOLO格式的相对坐标
                    x_center = (x + w/2) / width
                    y_center = (y + h/2) / height
                    w = w / width
                    h = h / height
                    
                    f.write(f"{ann['category_id']} {x_center} {y_center} {w} {h}\n")

    def _save_coco(self, dataset: Dict, output_path: str):
        """保存为COCO格式"""
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

    def _save_voc(self, dataset: Dict, output_dir: str):
        """保存为VOC格式"""
        os.makedirs(output_dir, exist_ok=True)
        
        for img in dataset['images']:
            img_id = img['id']
            
            # 创建XML根节点
            root = ET.Element('annotation')
            
            # 添加基本信息
            ET.SubElement(root, 'filename').text = img['file_name']
            
            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = str(img['width'])
            ET.SubElement(size, 'height').text = str(img['height'])
            ET.SubElement(size, 'depth').text = '3'
            
            # 添加标注信息
            annotations = [ann for ann in dataset['annotations'] if ann['image_id'] == img_id]
            
            for ann in annotations:
                obj = ET.SubElement(root, 'object')
                
                category = next(cat for cat in dataset['categories'] if cat['id'] == ann['category_id'])
                ET.SubElement(obj, 'name').text = category['name']
                ET.SubElement(obj, 'pose').text = 'Unspecified'
                ET.SubElement(obj, 'truncated').text = '0'
                ET.SubElement(obj, 'difficult').text = '0'
                
                bbox = ET.SubElement(obj, 'bndbox')
                x, y, w, h = ann['bbox']
                ET.SubElement(bbox, 'xmin').text = str(int(x))
                ET.SubElement(bbox, 'ymin').text = str(int(y))
                ET.SubElement(bbox, 'xmax').text = str(int(x + w))
                ET.SubElement(bbox, 'ymax').text = str(int(y + h))
            
            # 保存XML文件
            tree = ET.ElementTree(root)
            xml_path = os.path.join(output_dir, os.path.splitext(img['file_name'])[0] + '.xml')
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)

if __name__ == '__main__':
    # 使用示例
    converter = HBBDatasetConverter()
    
    # YOLO转COCO
    converter.convert(
        source_dir="path/to/yolo/dataset",
        source_format="yolo",
        target_format="coco",
        output_path="output.json",
        classes_file="path/to/classes.txt"
    )
    
    # COCO转VOC
    converter.convert(
        source_dir="path/to/coco.json",
        source_format="coco",
        target_format="voc",
        output_path="output_voc_dir"
    )
    
    # VOC转YOLO
    converter.convert(
        source_dir="path/to/voc/dataset",
        source_format="voc",
        target_format="yolo",
        output_path="output_yolo_dir"
    ) 