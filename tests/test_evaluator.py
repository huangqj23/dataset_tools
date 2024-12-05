import os
from pathlib import Path
from dataset_tools.hbb import HBBDatasetEvaluator

def test_evaluation():
    # 获取测试数据路径
    current_dir = Path(__file__).parent
    gt_path = current_dir / 'data/ground_truth'
    pred_path = current_dir / 'data/predictions'
    classes_file = current_dir / 'data/classes.txt'
    output_dir = current_dir / 'output/evaluation'
    
    # 创建评估器
    evaluator = HBBDatasetEvaluator()
    
    # 运行评估
    evaluator.evaluate(
        gt_path=gt_path,
        pred_path=pred_path,
        gt_format='yolo',
        pred_format='yolo',
        output_dir=output_dir,
        classes_file=classes_file
    )
    
    # 检查输出文件是否存在
    assert (output_dir / 'evaluation_results.json').exists()
    assert (output_dir / 'pr_curves.png').exists()

if __name__ == '__main__':
    test_evaluation() 