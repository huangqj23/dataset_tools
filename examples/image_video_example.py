from dataset_tools.utils.video_image_converter import VideoImageConverter

if __name__ == '__main__':
    # 创建转换器实例
    converter = VideoImageConverter()
    
    # 图片转视频示例
    converter.images_to_video(
        image_dir='/data1/huangqj/images/',
        output_path='output.mp4',
        fps=30,
        image_format='.JPG',
        frames_per_image=30,
        num_workers=8
    )
    