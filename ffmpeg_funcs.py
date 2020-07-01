
import os
import subprocess


def ffmpeg_grab_frames(source_path, frame_nums, output_directory, dest_width=1280, dest_height=720, filename_prefix="frame_", fps=30/1001):
    if not os.path.isdir(output_directory):
        raise Exception(
            'output_directory must be a directory: {}'.format(output_directory))

    combined_prefix = os.path.join(output_directory, filename_prefix)

    eq_expressions = ""
    output_files = []
    for idx, frame_num in enumerate(frame_nums):
        output_files.append(filename_prefix + str(idx + 1) + ".jpg")
        eq_expressions += "eq(n\\," + str(frame_num) + ")"
        if idx < len(frame_nums) - 1:
            eq_expressions += "+"

    args = ["ffmpeg", "-y", "-i", source_path, "-vf",
            "fps=" + str(fps) + ",select='" + eq_expressions + "',scale=" + str(dest_width) + ":" + str(dest_height), "-vsync", "0",  combined_prefix + "%d.jpg"]

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    result.check_returncode()

    return output_files


def ffmpeg_htile_images(framepaths, output_path):

    inputs = ["ffmpeg", "-y"]
    for framepath in framepaths:
        inputs.append("-i")
        inputs.append(framepath)

    args = inputs + ["-an", "-filter_complex",
                     "hstack=inputs=" + str(len(framepaths)), "-vsync", "0", "-frames:v", "1", output_path]

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    if result.returncode:
        print(result.stderr)

    result.check_returncode()
