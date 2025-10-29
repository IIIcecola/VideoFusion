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


def check_environment():
    """检查并返回当前运行环境信息"""
    return {
        "python_path": sys.executable,
        "conda_env": os.environ.get('CONDA_DEFAULT_ENV', 'None')
    }

def crop_video(input_path: str, output_path: str, rect: tuple[int, int, int, int]):
    """使用FFmpeg裁剪视频"""
    x, y, w, h = rect
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
        print(f"裁剪完成: {input_path} -> {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"裁剪失败{input_path}: {e.stderr}")
    except FileNotFoundError:
        print("未找到ffmpeg，请确保已安装并添加到环境变量")


def batch_crop_videos(input_dir: str, output_dir: str) -> str:
    """递归处理目录下所有视频，保持目录结构"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_extensions = ('.mp4', '.avi', '.flv', '.mov', '.mkv')

    # 递归遍历所有视频文件
    for input_path in input_dir.rglob('*'):
        if input_path.suffix.lower() in video_extensions and input_path.is_file():
            # 构建输出路径（保持目录结构+添加_noblack后缀）
            rel_path = input_path.relative_to(input_dir)
            output_file = output_dir / rel_path.parent / f"{rel_path.stem}_noblack{rel_path.suffix}"
            
            # 跳过已处理文件
            if output_file.exists():
                print(f"已处理，跳过: {input_path}")
                continue

            # 检测黑边区域
            remover = VideoRemover()
            max_rect = remover.remove_black(str(input_path))

            # 处理结果判断
            if max_rect[2] <= 0 or max_rect[3] <= 0:  # 无效视频
                print(f"跳过损坏视频: {input_path}")
                continue
            if max_rect[2] == input_path.stat().st_size and max_rect[3] == input_path.stat().st_size:  # 无黑边
                output_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_path, output_file)
                print(f"无黑边，复制完成: {input_path} -> {output_file}")
            else:  # 需要裁剪
                output_file.parent.mkdir(parents=True, exist_ok=True)
                crop_video(str(input_path), str(output_file), max_rect)

    return str(output_dir)


def main():
    """管线适配主函数：处理JSON输入输出"""
    result = {
        "success": False,
        "video_path": None, # 为方便模块之间调用，输出路径统一命名为'video_path'
        "environment_info": None,
        "error": None
    }

    try:
        # 解析命令行参数
        if len(sys.argv) != 3:
            raise ValueError("用法: python video_black_remover.py <input_file> <output_file>")
        input_file = sys.argv[1]
        output_file = sys.argv[2]

        # 校验输入文件
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")

        # 读取JSON配置
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        # 提取参数
        video_path = input_data['params'].get('video_path').get('video_path')
        output_path = input_data['config'].get('output_path', 'output_noblack')
        if not video_path:
            raise KeyError("输入JSON缺少'video_path'参数")

        # 执行处理
        result["video_path"] = batch_crop_videos(video_path, output_path)
        result["environment_info"] = check_environment()
        result["success"] = True
        print("批量处理完成")

    except Exception as e:
        result["error"] = str(e)
        print(f"处理失败: {e}")

    finally:
        # 写入输出结果
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as write_err:
            print(f"写入输出文件失败: {write_err}")
            sys.exit(1)

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
