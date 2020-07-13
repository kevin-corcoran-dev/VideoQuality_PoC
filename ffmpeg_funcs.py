
import os
import subprocess
import numpy as np
import re
from fractions import Fraction


def grab_thumbnail_strip(source_path, output_dir, duration, fps, steps):

    if not os.path.isdir(output_dir):
        raise Exception(
            'output_directory must be a directory: {}'.format(output_dir))

    output_file = os.path.basename(source_path)
    output_file = os.path.splitext(output_file)[0]

    output_file = os.path.join(output_dir, output_file + "_thumbnails.jpg")

    times = np.linspace(0, duration - 0.1, steps)
    frame_nums = [int(time * fps) for time in times]

    output_frames = ffmpeg_grab_frames(
        source_path, frame_nums, output_dir,  214, 120, "frame_", fps)

    # ffmpeg_funcs.ffmpeg_grab_frames() returns relative paths -- let's turn them into absolut
    output_frames = [os.path.join(output_dir, output_frame)
                     for output_frame in output_frames]

    ffmpeg_htile_images(output_frames, output_file)

    # we've got the strip, let's delete the component files
    for file in output_frames:
        os.remove(file)

    return output_file


def ffprobe_get_sample_aspect_ratio(source):
    result = subprocess.run(["ffprobe", "-select_streams", "v:0", "-show_entries", "stream=sample_aspect_ratio", "-of", "default=noprint_wrappers=1:nokey=1", source],
                            shell=False, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, )

    sample_aspect_ratio = result.stdout
    if sample_aspect_ratio == "N/A\n":
        sample_aspect_ratio = "1:1"

    return sample_aspect_ratio


def ffprobe_get_video_duration(source):
    result = subprocess.run(["ffprobe", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1", source],
                            shell=False, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, )

    return float(result.stdout)


def ffprobe_get_audio_duration(source):
    result = subprocess.run(["ffprobe", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1", source],
                            shell=False, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, )

    return float(result.stdout)


def ffprobe_get_framerate(source):
    result = subprocess.run(["ffprobe", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", source],
                            shell=False, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, )

    return float(Fraction(result.stdout))


def ffmpeg_get_deinterlace(source, duration):
    result = subprocess.run(["ffmpeg", "-i", source,   "-ss",
                             str(duration / 2), "-vf", "idet", "-vframes",
                             "200", "-an", "-f", "NULL", "-"],
                            shell=False, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, )

    tff_match = re.search(r'(?:TFF: )([0-9. ]*)', result.stderr)
    tff = int(tff_match.group(1))

    bff_match = re.search(r'(?:BFF: )([0-9. ]*)', result.stderr)
    bff = int(bff_match.group(1))

    interlaced_frames = tff + bff
    if interlaced_frames > 70:
        return "yadif"
    else:
        return ""


def ffprobe_get_frame_size(source):
    result = subprocess.run(["ffprobe", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "default=noprint_wrappers=1", source],
                            shell=False, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, )

    return {line.split('=')[0]: int(line.split('=')[1]) for line in result.stdout.split()}


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


def generate_diff_movie(source, transcode, output_path, dest_fps, target_width, target_height):

    if not output_path.endswith(".mp4"):
        raise Exception(
            'log_path must end in .mp4: {}'.format(output_path))

    args = ["ffmpeg", "-r", str(dest_fps), "-i", source, "-r", str(dest_fps), "-i", transcode, "-an",
            "-filter_complex", "[0:v]setpts=PTS-STARTPTS,fps=" + str(dest_fps) + ",scale=" + str(target_width) + ":" + str(target_height) + "[source];[1:v]setpts=PTS-STARTPTS[trancoded];[source][trancoded]blend=all_mode=difference,hue=s=0,eq=gamma=1.7", output_path]

    joined_args = " ".join(args)
    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    result.check_returncode()


def transcode(source_path, video_template, output_path):

    video_template = video_template.split()
    args = ["ffmpeg", "-y", "-i", source_path, ] + \
        video_template + [output_path]

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    if result.returncode:
        print(result.stderr)

    result.check_returncode()
