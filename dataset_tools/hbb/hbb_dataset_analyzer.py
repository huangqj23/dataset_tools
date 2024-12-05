import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

class HBBDatasetAnalyzer:
    """水平框数据集分析工具，支持对数据集进行多维度统计分析"""
    
    def __init__(self):
        self.supported_formats = ['yolo', 'coco', 'voc']
        
    def analyze(self,
               image_path: Union[str, Path],
               label_path: Union[str, Path],
               format: str,
               output_dir: Union[str, Path],
               classes_file: Optional[Union[str, Path]] = None,
               num_workers: int = 4):
        """
        分析数据集统计信息
        
        Args:
            image_path: 图片目录
            label_path: 标注文件路径或标注目录
            format: 数据集格式
            output_dir: 统计结果保存目录
            classes_file: YOLO格式的类别文件路径
            num_workers: 并行处理的线程数
        """
        image_path = Path(image_path)
        label_path = Path(label_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集数据
        stats = self._collect_statistics(
            image_path, label_path, format, classes_file, num_workers
        )
        
        # 生成统计图
        self._plot_image_size_distribution(stats['image_sizes'], output_dir)
        self._plot_bbox_size_distribution(stats['bbox_sizes'], output_dir)
        self._plot_aspect_ratio_distribution(stats['aspect_ratios'], output_dir)
        self._plot_bbox_clusters(stats['bbox_sizes'], output_dir)
        self._plot_category_distribution(stats['categories'], output_dir)
        
        # 保存统计结果
        self._save_statistics(stats, output_dir)
    
    def _collect_statistics(self,
                          image_path: Path,
                          label_path: Path,
                          format: str,
                          classes_file: Optional[Path],
                          num_workers: int) -> Dict:
        """收集数据集统计信息"""
        # 初始化统计数据
        stats = {
            'image_sizes': [],    # [(width, height), ...]
            'bbox_sizes': [],     # [(width, height), ...]
            'aspect_ratios': [],  # [w/h, ...]
            'categories': {}      # {category: count}
        }
        
        # 收集图片文件
        image_patterns = [
            "*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]",
            "*.[pP][nN][gG]", "*.[tT][iI][fF]", "*.[tT][iI][fF][fF]"
        ]
        
        image_files = []
        for pattern in image_patterns:
            image_files.extend(list(image_path.glob(pattern)))
            
        # 读取YOLO类别文件
        if format == 'yolo' and classes_file:
            with open(classes_file, 'r') as f:
                classes = f.read().strip().split('\n')
        
        # 处理单个图片
        def process_image(img_file: Path) -> Tuple[List, List, List, Dict]:
            try:
                # 读取图片尺寸
                img = cv2.imread(str(img_file))
                if img is None:
                    return [], [], [], {}
                height, width = img.shape[:2]
                
                # 读取标注
                if format == 'voc':
                    boxes, labels = self._parse_voc(label_path / f"{img_file.stem}.xml")
                elif format == 'yolo':
                    boxes, labels = self._parse_yolo(
                        label_path / f"{img_file.stem}.txt",
                        width, height, classes
                    )
                else:  # coco
                    boxes, labels = self._parse_coco(label_path, img_file.name)
                
                # 收集统计信息
                img_sizes = [(width, height)]
                bbox_sizes = []
                aspect_ratios = []
                category_counts = {}
                
                for box, label in zip(boxes, labels):
                    x1, y1, x2, y2 = map(float, box)
                    w = x2 - x1
                    h = y2 - y1
                    bbox_sizes.append((w, h))
                    aspect_ratios.append(w / h if h > 0 else 0)
                    category_counts[label] = category_counts.get(label, 0) + 1
                
                return img_sizes, bbox_sizes, aspect_ratios, category_counts
                
            except Exception as e:
                print(f"处理 {img_file.name} 时出错: {str(e)}")
                return [], [], [], {}
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_image, image_files),
                total=len(image_files),
                desc="Collecting statistics"
            ))
        
        # 合并结果
        for img_sizes, bbox_sizes, aspect_ratios, category_counts in results:
            stats['image_sizes'].extend(img_sizes)
            stats['bbox_sizes'].extend(bbox_sizes)
            stats['aspect_ratios'].extend(aspect_ratios)
            for cat, count in category_counts.items():
                stats['categories'][cat] = stats['categories'].get(cat, 0) + count
        
        return stats
    
    def _plot_image_size_distribution(self, sizes: List[Tuple[int, int]], output_dir: Path):
        """绘制图片尺寸分布图"""
        plt.figure(figsize=(15, 5))
        
        # 宽度分布
        plt.subplot(131)
        widths = [w for w, _ in sizes]
        sns.histplot(widths, bins=50)
        plt.title('Image Width Distribution')
        plt.xlabel('Width')
        
        # 高度分布
        plt.subplot(132)
        heights = [h for _, h in sizes]
        sns.histplot(heights, bins=50)
        plt.title('Image Height Distribution')
        plt.xlabel('Height')
        
        # 散点图
        plt.subplot(133)
        plt.scatter([w for w, _ in sizes], [h for _, h in sizes], alpha=0.5)
        plt.title('Image Size Distribution')
        plt.xlabel('Width')
        plt.ylabel('Height')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'image_size_distribution.png')
        plt.close()
    
    def _plot_bbox_size_distribution(self, sizes: List[Tuple[float, float]], output_dir: Path):
        """绘制边界框尺寸分布图"""
        plt.figure(figsize=(15, 5))
        
        # 宽度分布
        plt.subplot(131)
        widths = [w for w, _ in sizes]
        sns.histplot(widths, bins=50)
        plt.title('BBox Width Distribution')
        plt.xlabel('Width')
        
        # 高度分布
        plt.subplot(132)
        heights = [h for _, h in sizes]
        sns.histplot(heights, bins=50)
        plt.title('BBox Height Distribution')
        plt.xlabel('Height')
        
        # 散点图
        plt.subplot(133)
        plt.scatter([w for w, _ in sizes], [h for _, h in sizes], alpha=0.5)
        plt.title('BBox Size Distribution')
        plt.xlabel('Width')
        plt.ylabel('Height')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'bbox_size_distribution.png')
        plt.close()
    
    def _plot_aspect_ratio_distribution(self, ratios: List[float], output_dir: Path):
        """绘制宽高比分布图"""
        plt.figure(figsize=(10, 5))
        
        # 过滤掉异常值
        filtered_ratios = [r for r in ratios if 0.1 <= r <= 10]
        
        # 直方图
        plt.subplot(121)
        sns.histplot(filtered_ratios, bins=50)
        plt.title('Aspect Ratio Distribution')
        plt.xlabel('Width/Height')
        
        # 箱型图
        plt.subplot(122)
        sns.boxplot(y=filtered_ratios)
        plt.title('Aspect Ratio Boxplot')
        plt.ylabel('Width/Height')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'aspect_ratio_distribution.png')
        plt.close()
    
    def _plot_bbox_clusters(self, sizes: List[Tuple[float, float]], output_dir: Path):
        """绘制边界框聚类图"""
        if not sizes:
            return
            
        # 转换为numpy数组
        X = np.array(sizes)
        
        # 使用K-means聚类
        n_clusters = min(5, len(X))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # 绘制聚类结果
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        plt.title('BBox Size Clusters')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.legend()
        plt.colorbar(scatter)
        
        plt.savefig(output_dir / 'bbox_clusters.png')
        plt.close()
    
    def _plot_category_distribution(self, categories: Dict[str, int], output_dir: Path):
        """绘制类别分布图"""
        plt.figure(figsize=(15, 5))
        
        # 条形图
        plt.subplot(121)
        categories_sorted = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))
        plt.bar(categories_sorted.keys(), categories_sorted.values())
        plt.xticks(rotation=45, ha='right')
        plt.title('Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('Count')
        
        # 饼图
        plt.subplot(122)
        plt.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%')
        plt.title('Category Proportion')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'category_distribution.png')
        plt.close()
    
    def _save_statistics(self, stats: Dict, output_dir: Path):
        """保存统计结果"""
        summary = {
            'image_count': len(stats['image_sizes']),
            'bbox_count': len(stats['bbox_sizes']),
            'category_count': len(stats['categories']),
            'image_size': {
                'width': {
                    'min': min(w for w, _ in stats['image_sizes']),
                    'max': max(w for w, _ in stats['image_sizes']),
                    'mean': np.mean([w for w, _ in stats['image_sizes']])
                },
                'height': {
                    'min': min(h for _, h in stats['image_sizes']),
                    'max': max(h for _, h in stats['image_sizes']),
                    'mean': np.mean([h for _, h in stats['image_sizes']])
                }
            },
            'bbox_size': {
                'width': {
                    'min': min(w for w, _ in stats['bbox_sizes']),
                    'max': max(w for w, _ in stats['bbox_sizes']),
                    'mean': np.mean([w for w, _ in stats['bbox_sizes']])
                },
                'height': {
                    'min': min(h for _, h in stats['bbox_sizes']),
                    'max': max(h for _, h in stats['bbox_sizes']),
                    'mean': np.mean([h for _, h in stats['bbox_sizes']])
                }
            },
            'aspect_ratio': {
                'min': min(stats['aspect_ratios']),
                'max': max(stats['aspect_ratios']),
                'mean': np.mean(stats['aspect_ratios'])
            },
            'categories': stats['categories']
        }
        
        with open(output_dir / 'statistics.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
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
            
        cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        image_id = None
        for img in coco_data['images']:
            if img['file_name'] == image_name:
                image_id = img['id']
                break
                
        if image_id is None:
            return [], []
            
        boxes = []
        labels = []
        
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
    analyzer = HBBDatasetAnalyzer()
    
    analyzer.analyze(
        image_path='datasets/images',
        label_path='datasets/labels',
        format='yolo',
        output_dir='output/analysis',
        classes_file='datasets/classes.txt'
    ) 