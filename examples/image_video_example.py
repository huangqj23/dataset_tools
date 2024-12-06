from dataset_tools.utils.video_image_converter import VideoImageConverter

if __name__ == '__main__':
    # 创建转换器实例
    converter = VideoImageConverter()
    
    # 图片转视频示例
    # converter.images_to_video(
    #     image_dir='/data1/huangqj/images/',
    #     output_path='output.mp4',
    #     fps=30,
    #     resize=(1920, 1080),
    #     image_format='.JPG',
    #     frames_per_image=30,
    #     transition_frames=15,
    #     num_workers=8
    # )
    
    # 视频转图片
    converter.video_to_images(
        video_path='output.mp4',
        output_dir='output_frames',
        frame_interval=30,  # 每5帧提取一帧
        image_format='.jpg',
        image_quality=95,
        resize=(1920, 1080)  # 可选的尺寸调整
    )