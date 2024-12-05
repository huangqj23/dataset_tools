from dataset_tools import OBBDatasetConverter, OBBDatasetVisualizer

def main():
    # 创建转换器实例
    converter = OBBDatasetConverter()
    visualizer = OBBDatasetVisualizer()

    # 设置类别列表
    categories = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 
                'tennis-court', 'basketball-court', 'ground-track-field', 
                'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 
                'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']
    
    converter.set_categories(categories)
    visualizer.set_class_names(categories)

    # 示例：DOTA转COCO
    converter.dota_to_coco(
        dota_dir='path/to/dota/dataset',
        output_path='path/to/output/annotations.json'
    )

    # 示例：可视化DOTA格式数据
    visualizer.visualize(
        image_path='path/to/image.png',
        label_path='path/to/label.txt',
        format='dota'
    )

if __name__ == '__main__':
    main() 