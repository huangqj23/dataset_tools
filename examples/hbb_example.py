from dataset_tools.hbb.hbb_dataset_visualizer import HBBDatasetVisualizer
from dataset_tools.hbb.hbb_dataset_cropper import HBBDatasetCropper

if __name__ == '__main__':
    
    # MAR20数据集
    # 定义飞机型号映射字典
    mar20_aircraft_mapping = {
        'A1': 'SU-35',
        'A2': 'C-130',
        'A3': 'C-17',
        'A4': 'C-5',
        'A5': 'F-16',
        'A6': 'TU160',
        'A7': 'E-3',
        'A8': 'B-52',
        'A9': 'P-3C',
        'A10': 'B-1B',
        'A11': 'E-8',
        'A12': 'TU-22',
        'A13': 'F-15',
        'A14': 'KC-135',
        'A15': 'F-22',
        'A16': 'FA-18',
        'A17': 'TU-95',
        'A18': 'KC-10',
        'A19': 'SU-34',
        'A20': 'SU-24'
    }
    
    # vis
    # visualizer = HBBDatasetVisualizer()
    
    # # 设置类别名称（对YOLO格式必需，其他格式可选）
    # class_names = list(mar20_aircraft_mapping.values())
    # visualizer.set_class_names(class_names)

    # # 可视化单张图片
    # visualizer.visualize(
        # image_path='/data1/DATA_126/hqj/MAR20/JPEGImages/',
        # label_path='/data1/DATA_126/hqj/MAR20/Annotations/hbb/',
        # format='voc',
        # save_dir='/data1/DATA_126/hqj/MAR20/vis/',
    #     num_workers=10
    # )

    # crop
    # 使用示例
    cropper = HBBDatasetCropper()
    
    # VOC格式
    cropper.crop(
        image_path='/data1/DATA_126/hqj/MAR20/JPEGImages/',
        label_path='/data1/DATA_126/hqj/MAR20/Annotations/hbb/',
        output_dir='/data1/DATA_126/hqj/MAR20/crops/',
        format='voc',
        num_workers=10
    )
    
    