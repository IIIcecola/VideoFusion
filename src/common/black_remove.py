import os
import sys
import json
import shutil
import subprocess
import cv2
import numpy as np
from pathlib import Path

project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
from src.common.black_remove_algorithm.black_remove_algorithm import BlackRemoveAlgorithm
from src.common.black_remove_algorithm.video_remover import VideoRemover
from src.common.black_remove_algorithm.img_black_remover import IMGBlackRemover
from src.common.black_remove.img_black_remover import BlackRemover 


def check_environment():
    """检查并返回当前运行环境信息"""
    return {
        "python_path": sys.executable,
        "conda_env": os.environ.get('CONDA_DEFAULT_ENV', 'None')
    }


def crop_video(input_path: str, output_path: str, rect: tuple[int, int, int, int]):
    """使用FFmpeg裁剪视频"""
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        raise ValueError(f"无效的裁剪参数: {rect}")
    
    command = [
        'ffmpeg',
        '-i', input_path,
        '-vf', f'crop={w}:{h}:{x}:{y}',
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'medium',
        '-c:a', 'copy',
        '-y',
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"视频裁剪完成: {input_path} -> {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"视频裁剪失败 {input_path}: {e.stderr}")
    except FileNotFoundError:
        print("未找到ffmpeg，请确保已安装并添加到环境变量")


def crop_image(input_path: str, output_path: str, rect: tuple[int, int, int, int]):
    """使用OpenCV裁剪图片"""
    x1, y1, x2, y2 = rect  # 图片处理返回的是左上角和右下角坐标
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        raise ValueError(f"无效的裁剪参数: {rect}")
    
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {input_path}")
    
    cropped_img = img[y1:y2, x1:x2]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cropped_img)
    print(f"图片裁剪完成: {input_path} -> {output_path}")


def get_remover(media_type: str, algorithm: str = "dynamic") -> BlackRemoveAlgorithm | BlackRemover:
    """根据媒体类型和算法选择对应的黑边处理器"""
    if media_type == "video":
        if algorithm == "dynamic":
            return VideoRemover()
        elif algorithm == "static":
            return IMGBlackRemover()
        else:
            raise ValueError(f"不支持的视频算法: {algorithm}")
    elif media_type == "image":
        return BlackRemover()  # 图片固定使用静态算法
    else:
        raise ValueError(f"不支持的媒体类型: {media_type}")


def batch_process_media(input_dir: str, output_dir: str, crop_enabled: bool = True, 
                       video_algorithm: str = "dynamic", max_frames: int = 500) -> str:
    """
    批量处理目录下所有图片和视频，保持目录结构
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    :param crop_enabled: 是否启用裁剪
    :param video_algorithm: 视频处理算法（dynamic/static）
    :param max_frames: 视频静态算法最大采样帧数
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 支持的媒体类型扩展名
    video_extensions = ('.mp4', '.avi', '.flv', '.mov', '.mkv')
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    for input_path in input_dir.rglob('*'):
        if not input_path.is_file():
            continue
        
        # 判断媒体类型
        suffix = input_path.suffix.lower()
        if suffix in video_extensions:
            media_type = "video"
        elif suffix in image_extensions:
            media_type = "image"
        else:
            print(f"跳过不支持的文件: {input_path}")
            continue

        # 构建输出路径
        rel_path = input_path.relative_to(input_dir)
        output_file = output_dir / rel_path.parent / f"{rel_path.stem}_noblack{rel_path.suffix}"
        
        # 跳过已处理文件
        if output_file.exists():
            print(f"已处理，跳过: {input_path}")
            continue

        try:
            # 获取黑边处理器
            remover = get_remover(media_type, video_algorithm)
            
            # 检测黑边区域
            if media_type == "video":
                if video_algorithm == "static":
                    rect = remover.remove_black(str(input_path), max_frames=max_frames)
                else:
                    rect = remover.remove_black(str(input_path))
                # 视频处理器返回 (x, y, w, h)
                x, y, w, h = rect
                original_w = int(cv2.VideoCapture(str(input_path)).get(cv2.CAP_PROP_FRAME_WIDTH))
                original_h = int(cv2.VideoCapture(str(input_path)).get(cv2.CAP_PROP_FRAME_HEIGHT))
                has_black = not (w == original_w and h == original_h)
            else:  # 图片
                # 图片处理器返回 (x1, y1, x2, y2)
                rect = remover.start(img_path=str(input_path))
                x1, y1, x2, y2 = rect
                img = cv2.imread(str(input_path))
                original_h, original_w = img.shape[:2]
                has_black = not (x1 == 0 and y1 == 0 and x2 == original_w and y2 == original_h)

            # 根据配置决定是否裁剪
            if not crop_enabled or not has_black:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_path, output_file)
                print(f"无需裁剪，复制完成: {input_path} -> {output_file}")
            else:
                if media_type == "video":
                    crop_video(str(input_path), str(output_file), rect)
                else:
                    crop_image(str(input_path), str(output_file), rect)

        except Exception as e:
            print(f"处理文件 {input_path} 失败: {str(e)}")
            continue

    return str(output_dir)


def main():
    """统一处理图片和视频的黑边检测与裁剪"""
    result = {
        "success": False,
        "video_path": None,  # 统一输出路径键名
        "environment_info": None,
        "error": None
    }

    try:
        if len(sys.argv) != 3:
            raise ValueError("用法: python black_remove.py <input_file> <output_file>")
        input_file = sys.argv[1]
        output_file = sys.argv[2]

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")

        # 读取配置
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        # 提取参数
        media_path = input_data['params'].get('video_path', {}).get('video_path')  # 兼容原有键名

        # 配置参数
        output_path = input_data['config'].get('output_path', 'output_noblack')
        crop_enabled = input_data['config'].get('crop_enabled', True)  # 是否裁剪
        video_algorithm = input_data['config'].get('video_algorithm', 'dynamic')  # or "static"
        max_frames = input_data['config'].get('max_frames', 500)  # 视频静态算法采样帧数

        # 执行处理
        result["video_path"] = batch_process_media(
            media_path,
            output_path,
            crop_enabled=crop_enabled,
            video_algorithm=video_algorithm,
            max_frames=max_frames
        )
        result["environment_info"] = check_environment()
        result["success"] = True
        print("批量处理完成")

    except Exception as e:
        result["error"] = str(e)
        print(f"处理失败: {e}")

    finally:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as write_err:
            print(f"写入输出文件失败: {write_err}")
            sys.exit(1)

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
