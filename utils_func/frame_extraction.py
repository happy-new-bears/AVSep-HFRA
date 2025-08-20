import os
import subprocess

# 设置源文件夹路径和输出的帧文件夹路径
input_dir = "/bask/projects/j/jiaoj-rep-learn/Dataset/MUSIC/11solo_9duet/11solo_9duet_raw_video"
output_dir = "/bask/projects/j/jiaoj-rep-learn/Dataset/MUSIC/11solo_9duet/1fps_frames"
count = 0
# 遍历输入目录中的所有子目录和文件
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".mp4"):
            video_path = os.path.join(root, file)
            # 获取当前文件的相对路径并替换为输出路径
            relative_path = os.path.relpath(root, input_dir)
            output_folder = os.path.join(output_dir, relative_path, file[:-4])  # 去掉.mp4后缀
            
            # 创建输出文件夹
            os.makedirs(output_folder, exist_ok=True)
            
            # 生成ffmpeg命令来提取帧
            output_pattern = os.path.join(output_folder, "%06d.jpg")  # 000001.jpg格式
            ffmpeg_command = [
                "ffmpeg", "-i", video_path, "-vf", "fps=1", output_pattern
            ]
            count += 1
            # 执行ffmpeg命令
            subprocess.run(ffmpeg_command)
print(count)
print("帧提取完成！")
