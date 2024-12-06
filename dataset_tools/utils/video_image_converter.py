import cv2
import os
from pathlib import Path
from typing import Union, Optional, List, Tuple
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class VideoImageConverter:
    """视频和图片转换工具，支持批量图片转视频和视频拆分为图片"""
    
    def __init__(self):
        # 支持的视频格式
        self.supported_video_formats = [
            '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', 
            '.webm', '.m4v', '.mpeg', '.mpg', '.3gp'
        ]
        
        # 支持的图片格式
        self.supported_image_formats = [
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', 
            '.webp', '.gif', '.ppm', '.pgm'
        ]
        
        # 视频编解码器映射
        self.codec_map = {
            '.mp4': 'mp4v',
            '.avi': 'XVID',
            '.mov': 'mp4v',
            '.mkv': 'mp4v',
            '.flv': 'FLV1',
            '.wmv': 'WMV2',
            '.webm': 'VP80',
            '.m4v': 'mp4v',
            '.mpeg': 'MPEG',
            '.mpg': 'MPEG',
            '.3gp': 'mp4v'
        }
        
        # 添加默认线程数
        self.default_workers = min(32, os.cpu_count() * 2)  # 设置合理的默认线程数
    
    def _read_image(self, img_file: Path, resize: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """读取并处理单张图片"""
        try:
            frame = cv2.imread(str(img_file))
            if frame is None:
                print(f"警告: 无法读取图片 {img_file}")
                return None
                
            if resize:
                frame = cv2.resize(frame, resize)
            return frame
        except Exception as e:
            print(f"处理图片 {img_file} 时出错: {str(e)}")
            return None
    
    def images_to_video(self,
                       image_dir: Union[str, Path],
                       output_path: Union[str, Path],
                       fps: int = 30,
                       image_format: str = '.jpg',
                       sort_files: bool = True,
                       resize: Optional[Tuple[int, int]] = None,
                       num_workers: Optional[int] = None,
                       frames_per_image: int = 1,
                       transition_frames: int = 0) -> bool:
        """
        将图片序列转换为视频，支持控制每张图片的持续时间
        
        Args:
            image_dir: 图片目录路径
            output_path: 输出视频路径
            fps: 视频帧率
            image_format: 图片格式（例如：'.jpg'）
            sort_files: 是否对文件名进行排序
            resize: 调整大小，格式为(width, height)
            num_workers: 线程数
            frames_per_image: 每张图片持续的帧数
            transition_frames: 图片之间的过渡帧数（淡入淡出效果）
            
        Returns:
            bool: 转换是否成功
        """
        image_dir = Path(image_dir)
        output_path = Path(output_path)
        
        # 验证路径
        if not image_dir.is_dir():
            raise ValueError(f"图片目录不存在: {image_dir}")
            
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图片文件
        image_files = [f for f in image_dir.glob(f"*{image_format}")]
        if not image_files:
            raise ValueError(f"未找到{image_format}格式的图片文件")
            
        if sort_files:
            image_files.sort()
        
        # 读取第一张图片获取尺寸
        first_image = cv2.imread(str(image_files[0]))
        if first_image is None:
            raise ValueError(f"无法读取图片: {image_files[0]}")
            
        if resize:
            frame_size = resize
        else:
            frame_size = (first_image.shape[1], first_image.shape[0])
        
        # 获取视频编解码器
        ext = output_path.suffix.lower()
        if ext not in self.supported_video_formats:
            raise ValueError(f"不支持的视频格式: {ext}")
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec_map[ext])
        
        # 创建视频写入器
        out = cv2.VideoWriter(
            str(output_path), fourcc, fps, frame_size
        )
        
        num_workers = num_workers or self.default_workers
        
        # 使用线程池预加载图片
        frames = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {
                executor.submit(self._read_image, img_file, resize): img_file 
                for img_file in image_files
            }
            
            for future in tqdm(as_completed(future_to_file), 
                             total=len(image_files),
                             desc="Loading images"):
                img_file = future_to_file[future]
                try:
                    frame = future.result()
                    if frame is not None:
                        frames.append((img_file, frame))
                except Exception as e:
                    print(f"处理图片 {img_file} 时出错: {str(e)}")
        
        # 按文件名排序
        if sort_files:
            frames.sort(key=lambda x: x[0].name)
        
        # 写入视频
        try:
            total_frames = len(frames) * frames_per_image + (len(frames) - 1) * transition_frames
            with tqdm(total=total_frames, desc="Writing video") as pbar:
                for i, (_, current_frame) in enumerate(frames):
                    # 写入当前图片的帧
                    for _ in range(frames_per_image):
                        out.write(current_frame)
                        pbar.update(1)
                    
                    # 如果不是最后一张图片且有过渡帧，则创建过渡效果
                    if i < len(frames) - 1 and transition_frames > 0:
                        next_frame = frames[i + 1][1]
                        # 创建淡入淡出效果
                        for t in range(transition_frames):
                            alpha = t / transition_frames
                            blended_frame = cv2.addWeighted(
                                current_frame, 1 - alpha,
                                next_frame, alpha,
                                0
                            )
                            out.write(blended_frame)
                            pbar.update(1)
            
            return True
        except Exception as e:
            print(f"写入视频时出错: {str(e)}")
            return False
        finally:
            out.release()
    
    def _save_frame(self,
                    frame: np.ndarray,
                    output_file: Path,
                    resize: Optional[Tuple[int, int]],
                    img_params: List) -> bool:
        """保存单帧图片"""
        try:
            if resize:
                frame = cv2.resize(frame, resize)
            cv2.imwrite(str(output_file), frame, img_params)
            return True
        except Exception as e:
            print(f"保存图片 {output_file} 时出错: {str(e)}")
            return False
    
    def video_to_images(self,
                       video_path: Union[str, Path],
                       output_dir: Union[str, Path],
                       frame_interval: int = 1,
                       image_format: str = '.jpg',
                       image_quality: int = 95,
                       resize: Optional[Tuple[int, int]] = None,
                       start_frame: int = 0,
                       end_frame: Optional[int] = None,
                       num_workers: Optional[int] = None,
                       buffer_size: int = 30) -> bool:
        """
        将视频拆分为图片序列
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出图片目录
            frame_interval: 帧间隔
            image_format: 输出图片格式
            image_quality: 图片质量(1-100)，仅对jpg格式有效
            resize: 调整大小，格式为(width, height)
            start_frame: 起始帧
            end_frame: 结束帧，None表示到视频结尾
            num_workers: 线程数
            buffer_size: 缓冲区大小
            
        Returns:
            bool: 转换是否成功
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        
        # 验证路径
        if not video_path.is_file():
            raise ValueError(f"视频文件不存在: {video_path}")
            
        # 验证图片格式
        if image_format not in self.supported_image_formats:
            raise ValueError(f"不支持的图片格式: {image_format}")
            
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        num_workers = num_workers or self.default_workers
        
        try:
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # 设置结束帧
            if end_frame is None or end_frame > total_frames:
                end_frame = total_frames
                
            # 设置起始帧
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 准备参数
            if image_format.lower() in ['.jpg', '.jpeg']:
                img_params = [cv2.IMWRITE_JPEG_QUALITY, image_quality]
            elif image_format.lower() == '.png':
                img_params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, image_quality // 10)]
            else:
                img_params = []
            
            frame_buffer = []
            frame_count = start_frame
            saved_count = 0
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                
                pbar = tqdm(total=(end_frame - start_frame) // frame_interval,
                           desc="Extracting frames")
                
                while frame_count < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_interval == 0:
                        # 创建输出文件路径
                        output_file = output_dir / f"frame_{frame_count:06d}{image_format}"
                        
                        # 将帧添加到缓冲区
                        frame_buffer.append((frame.copy(), output_file))
                        
                        # 当缓冲区达到指定大小时，批量处理
                        if len(frame_buffer) >= buffer_size:
                            # 提交所有帧的保存任务
                            batch_futures = [
                                executor.submit(self._save_frame, f, p, resize, img_params)
                                for f, p in frame_buffer
                            ]
                            futures.extend(batch_futures)
                            
                            # 清空缓冲区
                            frame_buffer = []
                            
                            # 更新进度条
                            saved_count += len(batch_futures)
                            pbar.update(len(batch_futures))
                    
                    frame_count += 1
                
                # 处理���余的帧
                if frame_buffer:
                    batch_futures = [
                        executor.submit(self._save_frame, f, p, resize, img_params)
                        for f, p in frame_buffer
                    ]
                    futures.extend(batch_futures)
                    saved_count += len(batch_futures)
                    pbar.update(len(batch_futures))
                
                # 等待所有任务完成
                for future in as_completed(futures):
                    if not future.result():
                        print("警告: 部分图片保存失败")
                
                pbar.close()
                print(f"\n共提取了 {saved_count} 帧图片")
                return True
                
        except Exception as e:
            print(f"转换过程中出错: {str(e)}")
            return False
            
        finally:
            cap.release()
    
    def batch_process_videos(self,
                           video_dir: Union[str, Path],
                           output_dir: Union[str, Path],
                           frame_interval: int = 1,
                           image_format: str = '.jpg',
                           video_format: str = '.mp4',
                           num_workers: Optional[int] = None,
                           **kwargs) -> None:
        """
        批量处理视频目录
        
        Args:
            video_dir: 视频目录
            output_dir: 输出目录
            frame_interval: 帧间隔
            image_format: 输出图片格式
            video_format: 处理的视频格式
            num_workers: 线程数
            **kwargs: 传递给video_to_images的其他参数
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        
        if not video_dir.is_dir():
            raise ValueError(f"视频目录不存在: {video_dir}")
        
        # 获取所有视频文件
        video_files = []
        if video_format == '*':
            for fmt in self.supported_video_formats:
                video_files.extend(video_dir.glob(f"*{fmt}"))
        else:
            video_files = list(video_dir.glob(f"*{video_format}"))
        
        if not video_files:
            raise ValueError(f"未找到{video_format}格式的视频文件")
        
        num_workers = num_workers or self.default_workers
        kwargs['num_workers'] = num_workers
        
        # 使用线程池并行处理视频
        with ThreadPoolExecutor(max_workers=min(8, num_workers)) as executor:  # 限制并行视频数
            futures = []
            for video_file in video_files:
                video_output_dir = output_dir / video_file.stem
                future = executor.submit(
                    self.video_to_images,
                    video_path=video_file,
                    output_dir=video_output_dir,
                    frame_interval=frame_interval,
                    image_format=image_format,
                    **kwargs
                )
                futures.append((future, video_file))
            
            # 等待所有视频处理完成
            for future, video_file in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"处理视频 {video_file.name} 时出错: {str(e)}")
    
    def calculate_video_duration(self,
                           num_images: int,
                           fps: int,
                           frames_per_image: int,
                           transition_frames: int = 0) -> float:
        """
        计算生成视频的总时长（秒）
        
        Args:
            num_images: 图片数量
            fps: 视频帧率
            frames_per_image: 每张图片持续的帧数
            transition_frames: 过渡帧数
        
        Returns:
            float: 视频总时长（秒）
        """
        total_frames = (num_images * frames_per_image + 
                       (num_images - 1) * transition_frames)
        return total_frames / fps

if __name__ == '__main__':
    # 创建转换器实例
    converter = VideoImageConverter()
    
    # 图片转视频示例
    converter.images_to_video(
        image_dir='path/to/images',
        output_path='output.mp4',
        fps=30,
        image_format='.jpg',
        resize=(1920, 1080)  # 可选的尺寸调整
    )
    
    # 视频转图片示例
    converter.video_to_images(
        video_path='input.mp4',
        output_dir='output_frames',
        frame_interval=5,  # 每5帧提取一帧
        image_format='.jpg',
        image_quality=95,
        resize=(1280, 720)  # 可选的尺寸调整
    )
    
    # 批量处理视频示例
    converter.batch_process_videos(
        video_dir='videos',
        output_dir='output_frames',
        frame_interval=10,
        image_format='.jpg',
        video_format='*',  # 处理所有支持的视频格式
        image_quality=95,
        resize=(1280, 720)
    ) 