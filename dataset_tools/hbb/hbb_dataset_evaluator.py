import numpy as np
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import seaborn as sns

class HBBDatasetEvaluator:
    """水平框数据集评估工具，支持VOC和COCO评价指标"""
    
    def __init__(self):
        self.supported_formats = ['yolo', 'coco', 'voc']
        
    def evaluate(self,
                gt_path: Union[str, Path],
                pred_path: Union[str, Path],
                gt_format: str,
                pred_format: str,
                output_dir: Union[str, Path],
                classes_file: Optional[Union[str, Path]] = None,
                iou_thresholds: Optional[List[float]] = None,
                num_workers: int = 4):
        """
        评估目标检测结果
        
        Args:
            gt_path: 真值标注路径
            pred_path: 预测结果路径
            gt_format: 真值格式 ('yolo', 'coco', 'voc')
            pred_format: 预测格式 ('yolo', 'coco', 'voc')
            output_dir: 评估结果保存目录
            classes_file: YOLO格式的类别文件路径
            iou_thresholds: IoU阈值列表，默认[0.5, 0.55, ..., 0.95]
            num_workers: 并行处理的线程数
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.5, 1.0, 0.05)
        
        # 转换为COCO格式进行评估
        gt_coco = self._convert_to_coco(gt_path, gt_format, classes_file)
        pred_coco = self._convert_to_coco(pred_path, pred_format, classes_file)
        
        # 计算COCO指标
        coco_metrics = self._evaluate_coco(gt_coco, pred_coco)
        
        # 计算VOC指标
        voc_metrics = self._evaluate_voc(gt_coco, pred_coco, iou_thresholds)
        
        # 打印评估结果
        self._print_metrics(coco_metrics, voc_metrics)
        
        # 保存结果
        self._save_results(coco_metrics, voc_metrics, output_dir)
    
    def _convert_to_coco(self, 
                        path: Path,
                        format: str,
                        classes_file: Optional[Path]) -> Dict:
        """将不同格式转换为COCO格式"""
        if format == 'coco':
            with open(path, 'r') as f:
                return json.load(f)
        
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # 读取类别
        if format == 'yolo':
            if not classes_file:
                raise ValueError("YOLO格式需要提供classes.txt文件")
            with open(classes_file, 'r') as f:
                classes = f.read().strip().split('\n')
            categories = [{'id': i, 'name': name} for i, name in enumerate(classes)]
        else:  # VOC
            categories = self._get_voc_categories(path)
        coco_data['categories'] = categories
        
        # 创建类别名称到ID的映射
        cat_name_to_id = {cat['name']: cat['id'] for cat in categories}
        
        # 收集图片和标注
        def process_file(file_path: Path, img_id: int) -> Tuple[Dict, List[Dict]]:
            try:
                if format == 'voc':
                    return self._convert_voc_to_coco(file_path, img_id, cat_name_to_id)
                else:  # yolo
                    return self._convert_yolo_to_coco(file_path, img_id, cat_name_to_id)
            except Exception as e:
                print(f"处理 {file_path} 时出错: {str(e)}")
                return None, []
        
        # 并行处理
        files = list(Path(path).glob(f"*.{'xml' if format == 'voc' else 'txt'}"))
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(
                executor.map(lambda x: process_file(x[1], x[0]), enumerate(files)),
                total=len(files),
                desc="Converting to COCO format"
            ))
        
        # 合并结果
        ann_id = 1
        for img_info, annotations in results:
            if img_info:
                coco_data['images'].append(img_info)
                for ann in annotations:
                    ann['id'] = ann_id
                    coco_data['annotations'].append(ann)
                    ann_id += 1
        
        return coco_data
    
    def _evaluate_coco(self, gt_coco: Dict, pred_coco: Dict) -> Dict:
        """使用COCO评价指标进行评估"""
        # 创建COCO对象
        coco_gt = COCO()
        coco_gt.dataset = gt_coco
        coco_gt.createIndex()
        
        # 确保预测结果中包含必要的字段
        for ann in pred_coco['annotations']:
            if 'score' not in ann:
                ann['score'] = 1.0
            if 'segmentation' not in ann:
                ann['segmentation'] = []
            if 'id' not in ann:
                ann['id'] = ann['image_id'] * 100000 + ann.get('category_id', 0)
        
        # 创建临时JSON文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(pred_coco['annotations'], f)
            f.flush()
            try:
                coco_dt = coco_gt.loadRes(f.name)
            except Exception as e:
                print(f"COCO评估出错: {str(e)}")
                print("预测结果示例:", pred_coco['annotations'][0])
                raise
        
        # 创建评估器
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # 提取评估结果
        metrics = {
            'AP@0.5:0.95': coco_eval.stats[0],
            'AP@0.5': coco_eval.stats[1],
            'AP@0.75': coco_eval.stats[2],
            'AP_small': coco_eval.stats[3],
            'AP_medium': coco_eval.stats[4],
            'AP_large': coco_eval.stats[5],
            'AR_max1': coco_eval.stats[6],
            'AR_max10': coco_eval.stats[7],
            'AR_max100': coco_eval.stats[8],
            'precision': coco_eval.eval['precision'].tolist(),
            'recall': coco_eval.eval['recall'].tolist()
        }
        
        # 计算每个类别的AP
        category_ap = {}
        for idx, cat in enumerate(gt_coco['categories']):
            # AP@[0.5:0.95]
            ap = coco_eval.eval['precision'][:, :, idx, 0, -1].mean()
            # AP@0.5
            ap50 = coco_eval.eval['precision'][0, :, idx, 0, -1].mean()
            # AP@0.75
            ap75 = coco_eval.eval['precision'][5, :, idx, 0, -1].mean()
            
            category_ap[cat['name']] = {
                'AP@[0.5:0.95]': float(ap),
                'AP@0.5': float(ap50),
                'AP@0.75': float(ap75)
            }
        metrics['category_ap'] = category_ap
        
        return metrics
    
    def _evaluate_voc(self, 
                     gt_coco: Dict,
                     pred_coco: Dict,
                     iou_thresholds: List[float]) -> Dict:
        """使用VOC评价指标进行评估"""
        metrics = {
            'mAP': {},
            'AP_per_class': {},
            'precision_recall_curves': {}
        }
        
        # 按类别评估
        for cat in gt_coco['categories']:
            cat_id = cat['id']
            cat_name = cat['name']
            
            # 获取该类别的真值和预测框
            gt_boxes = self._get_boxes_by_category(gt_coco, cat_id)
            pred_boxes = self._get_boxes_by_category(pred_coco, cat_id)
            
            # 计算不同IoU阈值下的AP
            ap_values = []
            pr_curves = []
            
            for iou_thresh in iou_thresholds:
                precision, recall, ap = self._calculate_ap(
                    gt_boxes, pred_boxes, iou_thresh
                )
                ap_values.append(ap)
                pr_curves.append((precision, recall))
            
            metrics['AP_per_class'][cat_name] = np.mean(ap_values)
            metrics['precision_recall_curves'][cat_name] = pr_curves
        
        # 计算mAP
        metrics['mAP'] = np.mean(list(metrics['AP_per_class'].values()))
        
        return metrics
    
    def _print_metrics(self, coco_metrics: Dict, voc_metrics: Dict):
        """打印评估指标"""
        print("\n" + "="*80)
        print("COCO Metrics:")
        print("-"*80)
        metrics_to_print = [
            ('AP@0.5:0.95', 'AP@[0.5:0.95]'),
            ('AP@0.5', 'AP@0.5'),
            ('AP@0.75', 'AP@0.75'),
            ('AP_small', 'AP (small)'),
            ('AP_medium', 'AP (medium)'),
            ('AP_large', 'AP (large)'),
            ('AR_max1', 'AR@1'),
            ('AR_max10', 'AR@10'),
            ('AR_max100', 'AR@100')
        ]
        
        # 打印总体指标
        for key, name in metrics_to_print:
            value = coco_metrics[key]
            print(f"{name:<15}: {value:.4f}")
        
        # 打印每个类别的AP
        print("\nCOCO Per-class AP:")
        print("-"*80)
        print(f"{'Category':<15} {'AP@[0.5:0.95]':<15} {'AP@0.5':<15} {'AP@0.75':<15}")
        print("-"*80)
        
        if 'category_ap' in coco_metrics:
            # 按AP@[0.5:0.95]值降序排序
            sorted_ap = sorted(
                coco_metrics['category_ap'].items(), 
                key=lambda x: x[1]['AP@[0.5:0.95]'], 
                reverse=True
            )
            for class_name, ap_dict in sorted_ap:
                print(f"{class_name:<15} "
                      f"{ap_dict['AP@[0.5:0.95]']:>14.4f} "
                      f"{ap_dict['AP@0.5']:>14.4f} "
                      f"{ap_dict['AP@0.75']:>14.4f}")
        
        print("\n" + "="*50)
        print("VOC Metrics:")
        print("-"*50)
        print(f"mAP@0.5: {voc_metrics['mAP']:.4f}")
        
        print("\nVOC Per-class AP@0.5:")
        print("-"*50)
        # 按AP值降序排序
        sorted_ap = sorted(voc_metrics['AP_per_class'].items(), 
                          key=lambda x: x[1], reverse=True)
        for class_name, ap in sorted_ap:
            print(f"{class_name:<15}: {ap:.4f}")
    
    def _save_results(self,
                     coco_metrics: Dict,
                     voc_metrics: Dict,
                     output_dir: Path):
        """保存评估结果"""
        results = {
            'coco_metrics': {
                'overall': {
                    'AP@[0.5:0.95]': coco_metrics['AP@0.5:0.95'],
                    'AP@0.5': coco_metrics['AP@0.5'],
                    'AP@0.75': coco_metrics['AP@0.75'],
                    'AP (small)': coco_metrics['AP_small'],
                    'AP (medium)': coco_metrics['AP_medium'],
                    'AP (large)': coco_metrics['AP_large'],
                    'AR@1': coco_metrics['AR_max1'],
                    'AR@10': coco_metrics['AR_max10'],
                    'AR@100': coco_metrics['AR_max100']
                },
                'per_class': {}
            },
            'voc_metrics': {
                'overall': {
                    'mAP@0.5': voc_metrics['mAP']
                },
                'per_class': {}
            }
        }
        
        # 添加COCO每个类别的AP
        if 'category_ap' in coco_metrics:
            # 按AP@[0.5:0.95]降序排序
            sorted_ap = sorted(
                coco_metrics['category_ap'].items(),
                key=lambda x: x[1]['AP@[0.5:0.95]'],
                reverse=True
            )
            for class_name, ap_dict in sorted_ap:
                results['coco_metrics']['per_class'][class_name] = {
                    'AP@[0.5:0.95]': ap_dict['AP@[0.5:0.95]'],
                    'AP@0.5': ap_dict['AP@0.5'],
                    'AP@0.75': ap_dict['AP@0.75']
                }
        
        # 添加VOC每个类别的AP
        sorted_ap = sorted(
            voc_metrics['AP_per_class'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for class_name, ap in sorted_ap:
            results['voc_metrics']['per_class'][class_name] = {
                'AP@0.5': ap
            }
        
        # 保存为JSON文件
        with open(output_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存为易读的文本文件
        with open(output_dir / 'evaluation_results.txt', 'w', encoding='utf-8') as f:
            # COCO结果
            f.write("="*80 + "\n")
            f.write("COCO Metrics:\n")
            f.write("-"*80 + "\n")
            
            # COCO总体指标
            for name, value in results['coco_metrics']['overall'].items():
                f.write(f"{name:<15}: {value:.4f}\n")
            
            # COCO每类AP
            f.write("\nCOCO Per-class AP:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Category':<15} {'AP@[0.5:0.95]':<15} {'AP@0.5':<15} {'AP@0.75':<15}\n")
            f.write("-"*80 + "\n")
            
            for class_name, ap_dict in results['coco_metrics']['per_class'].items():
                f.write(f"{class_name:<15} "
                       f"{ap_dict['AP@[0.5:0.95]']:>14.4f} "
                       f"{ap_dict['AP@0.5']:>14.4f} "
                       f"{ap_dict['AP@0.75']:>14.4f}\n")
            
            # VOC结果
            f.write("\n" + "="*50 + "\n")
            f.write("VOC Metrics:\n")
            f.write("-"*50 + "\n")
            f.write(f"mAP@0.5: {results['voc_metrics']['overall']['mAP@0.5']:.4f}\n")
            
            # VOC每类AP
            f.write("\nVOC Per-class AP@0.5:\n")
            f.write("-"*50 + "\n")
            for class_name, ap_dict in results['voc_metrics']['per_class'].items():
                f.write(f"{class_name:<15}: {ap_dict['AP@0.5']:.4f}\n")
    
    def _get_voc_categories(self, voc_dir: Path) -> List[Dict]:
        """从VOC数据集中获取所有类别"""
        categories = set()
        for xml_file in Path(voc_dir).glob('*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                categories.add(obj.find('name').text)
        
        return [{'id': i, 'name': name} 
                for i, name in enumerate(sorted(categories))]
    
    def _convert_voc_to_coco(self,
                            xml_path: Path,
                            img_id: int,
                            cat_name_to_id: Dict) -> Tuple[Dict, List[Dict]]:
        """将VOC格式转换为COCO格式"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 片信息
        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        image_info = {
            'id': img_id,
            'file_name': filename,
            'width': width,
            'height': height
        }
        
        # 标注信息
        annotations = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            w = xmax - xmin
            h = ymax - ymin
            
            annotations.append({
                'image_id': img_id,
                'category_id': cat_name_to_id[name],
                'bbox': [xmin, ymin, w, h],
                'area': w * h,
                'iscrowd': 0
            })
        
        return image_info, annotations
    
    def _convert_yolo_to_coco(self,
                             txt_path: Path,
                             img_id: int,
                             cat_name_to_id: Dict) -> Tuple[Dict, List[Dict]]:
        """将YOLO格式转换为COCO格式"""
        # 获取对应的图片
        img_name = txt_path.stem + '.jpg'
        img_path = txt_path.parent.parent / 'images' / img_name
        if not img_path.exists():
            img_name = txt_path.stem + '.png'
            img_path = txt_path.parent.parent / 'images' / img_name
        
        if not img_path.exists():
            raise ValueError(f"找不到对应的图片文件: {img_path}")
        
        # 读取图片尺寸
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")
        height, width = img.shape[:2]
        
        image_info = {
            'id': img_id,
            'file_name': img_name,  # 只保存文件名
            'width': width,
            'height': height
        }
        
        # 读取标注
        annotations = []
        ann_id = img_id * 100000  # 使用更大的间隔避免ID冲突
        
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        try:
                            data = line.strip().split()
                            class_id = int(data[0])
                            x_center, y_center, w, h = map(float, data[1:5])
                            score = float(data[5]) if len(data) > 5 else 1.0
                            
                            # 转换为绝对坐标
                            w = w * width
                            h = h * height
                            x = x_center * width - w/2
                            y = y_center * height - h/2
                            
                            annotations.append({
                                'id': ann_id,
                                'image_id': img_id,
                                'category_id': class_id,
                                'bbox': [x, y, w, h],
                                'area': w * h,
                                'iscrowd': 0,
                                'score': score,
                                'segmentation': []
                            })
                            ann_id += 1
                        except Exception as e:
                            print(f"警告: 解行 '{line.strip()}' 时出错: {str(e)}")
                            continue
        except Exception as e:
            print(f"警告: 读取文件 {txt_path} 时出错: {str(e)}")
            return image_info, []
        
        return image_info, annotations
    
    def _get_boxes_by_category(self, coco_data: Dict, category_id: int) -> List[Dict]:
        """获取指定类别的所有边界框"""
        return [ann for ann in coco_data['annotations'] 
                if ann['category_id'] == category_id]
    
    def _calculate_ap(self,
                     gt_boxes: List[Dict],
                     pred_boxes: List[Dict],
                     iou_thresh: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """计算某个类别在特定IoU阈值下的AP值"""
        # 按置信度排序预测框
        pred_boxes = sorted(pred_boxes, key=lambda x: x.get('score', 1.0), reverse=True)
        
        # 初始化
        num_gt = len(gt_boxes)
        num_pred = len(pred_boxes)
        gt_matched = [False] * num_gt
        
        tp = np.zeros(num_pred)
        fp = np.zeros(num_pred)
        
        # 遍历每个预测框
        for pred_idx, pred_box in enumerate(pred_boxes):
            max_iou = 0
            max_idx = -1
            
            # 算与所有真值框的IoU
            for gt_idx, gt_box in enumerate(gt_boxes):
                if not gt_matched[gt_idx]:
                    iou = self._calculate_iou(pred_box['bbox'], gt_box['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = gt_idx
            
            # 判断是否匹配
            if max_iou >= iou_thresh:
                if not gt_matched[max_idx]:
                    tp[pred_idx] = 1
                    gt_matched[max_idx] = True
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # 计算累积值
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算precision和recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / num_gt if num_gt > 0 else np.zeros_like(tp_cumsum)
        
        # 计算AP
        ap = self._calculate_voc_ap(recall, precision)
        
        return precision, recall, ap
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """计算两边界框的IoU"""
        # COCO格式: [x,y,w,h]
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 计算交集
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_voc_ap(self, recall: np.ndarray, precision: np.ndarray) -> float:
        """使用VOC2007方法计算AP"""
        # 在开头添加哨兵
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
        
        # 计算precision包络
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        # 计算PR曲线下面积
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        return ap

if __name__ == '__main__':
    
    evaluator = HBBDatasetEvaluator()
    
    # 评估YOLO格式预测结果
    evaluator.evaluate(
        gt_path='datasets/ground_truth',
        pred_path='datasets/predictions',
        gt_format='yolo',
        pred_format='yolo',
        output_dir='output/evaluation',
        classes_file='datasets/classes.txt'
    ) 