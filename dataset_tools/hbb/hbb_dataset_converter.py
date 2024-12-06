import os
import json
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm
from glob import glob
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class HBBDatasetConverter:
    def __init__(self):
        self.supported_formats = ['yolo', 'coco', 'voc']
        
    def convert(self, source_dir: Union[str, Path], 
                source_format: str, 
                target_format: str, 
                output_path: Union[str, Path],
                classes_file: Optional[Union[str, Path]] = None):
        """
        转换数据集格式
        
        Args:
            source_dir: 源数据集目录(图片目录)
            source_format: 源格式 ('yolo', 'coco', 'voc')
            target_format: 目标格式 ('yolo', 'coco', 'voc')
            output_path: 输出路径(COCO格式为json文件路径，其他格式为输出目录)
            classes_file: 类别文件路径（YOLO格式需要）
        """
        source_dir = Path(source_dir)
        output_path = Path(output_path)
        if classes_file:
            classes_file = Path(classes_file)
        
        if source_format not in self.supported_formats or target_format not in self.supported_formats:
            raise ValueError(f"不支持的格式。支持的格式为: {self.supported_formats}")
        
        # 验证路径
        if not source_dir.is_dir():
            raise ValueError(f"源数据集目录不存在: {source_dir}")
        
        if source_format == 'yolo' and (not classes_file or not classes_file.is_file()):
            raise ValueError("YOLO格式需要提供classes.txt文件")
        
        # 读取数据集
        if source_format == 'yolo':
            dataset = self._read_yolo(source_dir, classes_file)
        elif source_format == 'coco':
            dataset = self._read_coco(source_dir)
        else:  # VOC
            dataset = self._read_voc(source_dir)
        
        # 转换并保存
        if target_format == 'yolo':
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True)
            self._save_yolo(dataset, output_path)
        elif target_format == 'coco':
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True)
            self._save_coco(dataset, output_path)
        else:  # VOC
            output_path.mkdir(parents=True, exist_ok=True)
            self._save_voc(dataset, output_path)

    def _read_yolo(self, yolo_dir: str, classes_file: str, num_workers: int = 4) -> Dict:
        """
        读取YOLO格式数据集
        
        Args:
            yolo_dir: YOLO数据集目录
            classes_file: 类别文件路径
            num_workers: 并行处理的线程数
        """
        dataset = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # 读取类别，ID从1开始
        with open(classes_file, 'r') as f:
            classes = f.read().strip().split('\n')
            for i, cls in enumerate(classes):
                dataset['categories'].append({
                    'id': i + 1,  # ID从1开始
                    'name': cls,
                    'supercategory': 'none'
                })
        
        # 收集所有图片文件
        image_patterns = [
            "*.[jJ][pP][gG]",      # jpg, JPG
            "*.[jJ][pP][eE][gG]",  # jpeg, JPEG
            "*.[pP][nN][gG]",      # png, PNG
            "*.[tT][iI][fF]",      # tif, TIF
            "*.[tT][iI][fF][fF]"   # tiff, TIFF
        ]
        
        img_files = []
        for pattern in image_patterns:
            img_files.extend(list(Path(yolo_dir).glob(pattern)))
            
        if not img_files:
            raise ValueError(f"未找到任何图片文件: {yolo_dir}")
            
        # 创建处理任务
        def process_image(img_id: int, img_path: Path) -> Tuple[Dict, List[Dict]]:
            try:
                # 读取图片信息
                img = cv2.imread(str(img_path))
                if img is None:
                    return None, []
                height, width = img.shape[:2]
                
                image_info = {
                    'id': img_id,
                    'file_name': img_path.name,
                    'width': width,
                    'height': height,
                    'path': str(img_path)
                }
                
                # 读取标注信息
                annotations = []
                txt_path = img_path.with_suffix('.txt')
                if txt_path.exists():
                    with open(txt_path, 'r') as f:
                        for line in f:
                            cls_id, x_center, y_center, w, h = map(float, line.strip().split())
                            
                            x = (x_center - w/2) * width
                            y = (y_center - h/2) * height
                            w = w * width
                            h = h * height
                            
                            annotations.append({
                                'image_id': img_id,
                                'category_id': int(cls_id) + 1,  # YOLO的类别ID加1
                                'bbox': [x, y, w, h],
                                'area': w * h,
                                'iscrowd': 0
                            })
                            
                return image_info, annotations
            except Exception as e:
                print(f"处理 {img_path.name} 时出错: {str(e)}")
                return None, []
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for img_id, img_path in enumerate(img_files):
                future = executor.submit(process_image, img_id, img_path)
                futures.append(future)
            
            # 收集结果
            ann_id = 0
            for future in tqdm(futures, desc="Processing images"):
                image_info, annotations = future.result()
                if image_info:
                    dataset['images'].append(image_info)
                    for ann in annotations:
                        ann['id'] = ann_id
                        dataset['annotations'].append(ann)
                        ann_id += 1
                        
        return dataset

    def _read_coco(self, coco_path: str) -> Dict:
        """��取COCO格式数据集"""
        with open(coco_path, 'r') as f:
            return json.load(f)

    def _read_voc(self, voc_dir: str, num_workers: int = 4) -> Dict:
        """
        读取VOC格式数据集
        
        Args:
            voc_dir: VOC数据集目录
            num_workers: 并行处理的线程数
        """
        dataset = {
            'images': [],
            'annotations': [],
            'categories': set()
        }
        
        # 收集所有XML文件
        xml_files = list(Path(voc_dir).glob("*.xml"))
        if not xml_files:
            raise ValueError(f"未找到任何XML文件: {voc_dir}")
            
        # 创建处理任务
        def process_xml(img_id: int, xml_file: Path) -> Tuple[Dict, List[Dict], List[str]]:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # 读取图片信息
                filename = root.find('filename').text
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                
                image_info = {
                    'id': img_id,
                    'file_name': filename,
                    'width': width,
                    'height': height,
                    'path': str(Path(voc_dir) / filename)
                }
                
                # 读取标注信息
                annotations = []
                categories = set()
                
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    categories.add(name)
                    
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    w = xmax - xmin
                    h = ymax - ymin
                    
                    annotations.append({
                        'image_id': img_id,
                        'category_name': name,
                        'bbox': [xmin, ymin, w, h],
                        'area': w * h,
                        'iscrowd': 0
                    })
                    
                return image_info, annotations, list(categories)
            except Exception as e:
                print(f"处理 {xml_file.name} 时出错: {str(e)}")
                return None, [], []
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for img_id, xml_file in enumerate(xml_files):
                future = executor.submit(process_xml, img_id, xml_file)
                futures.append(future)
            
            # 收集结果
            ann_id = 0
            for future in tqdm(futures, desc="Processing annotations"):
                image_info, annotations, categories = future.result()
                if image_info:
                    dataset['images'].append(image_info)
                    for ann in annotations:
                        ann['id'] = ann_id
                        dataset['annotations'].append(ann)
                        ann_id += 1
                    dataset['categories'].update(categories)
        
        # 转换categories为列表格式，ID从1开始
        categories = []
        for i, name in enumerate(sorted(dataset['categories'])):
            categories.append({
                'id': i + 1,  # ID从1开始
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
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存类别文件
        with open(output_dir / 'classes.txt', 'w') as f:
            # 按ID排序类别（减1后）
            categories = sorted(dataset['categories'], key=lambda x: x['id'])
            for cat in categories:
                f.write(f"{cat['name']}\n")
        
        # 创建类别ID映射（COCO ID -> YOLO ID）
        coco_to_yolo_id = {cat['id']: i for i, cat in enumerate(categories)}
        
        # 保存标注文件
        for img in dataset['images']:
            img_id = img['id']
            width = img['width']
            height = img['height']
            
            # 获取该图片的所有标注
            annotations = [ann for ann in dataset['annotations'] if ann['image_id'] == img_id]
            
            # 创建txt文件
            txt_path = output_dir / os.path.splitext(img['file_name'])[0] + '.txt'
            with open(txt_path, 'w') as f:
                for ann in annotations:
                    x, y, w, h = ann['bbox']
                    # 转换为YOLO格式的相对坐标
                    x_center = (x + w/2) / width
                    y_center = (y + h/2) / height
                    w = w / width
                    h = h / height
                    
                    # 将COCO类别ID转换为YOLO类别ID（从0开始）
                    yolo_cat_id = coco_to_yolo_id[ann['category_id']]
                    
                    f.write(f"{yolo_cat_id} {x_center} {y_center} {w} {h}\n")

    def _save_coco(self, dataset: Dict, output_path: str):
        """保存为COCO格式"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

    def _save_voc(self, dataset: Dict, output_dir: str):
        """保存为VOC格式"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for img in dataset['images']:
            img_id = img['id']
            
            # 创建XML根节点
            root = ET.Element('annotation')
            
            # 添加基本信
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
            xml_path = output_dir / os.path.splitext(img['file_name'])[0] + '.xml'
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