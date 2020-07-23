import sys
import os
import m3u8
from jinja2 import Environment, FileSystemLoader
from PIL import Image
from os.path import expanduser

import numpy as np
import ffmpeg_funcs as ffmpeg


def write_html_report(report_data, report_path):
    jinja_loader = FileSystemLoader('templates')
    jinja_env = Environment(loader=jinja_loader)
    jinja_temp = jinja_env.get_template("cue_points.html")
    output = jinja_temp.render(data=report_data)
    print(output)
    with open(report_path, "w") as f:
        f.write(output)


def grab_thumbnails_sequence(source_path, output_dir, cue_points, fps):

    results = []

    if not os.path.isdir(output_dir):
        raise Exception(
            'output_directory must be a directory: {}'.format(output_dir))

    cursor = Image.new('RGB', (10, 120), color='red')
    cursor.save(os.path.join(output_dir, 'cursor.jpg'))

    base_output_file = os.path.basename(source_path)
    base_output_file = os.path.splitext(base_output_file)[0]

    all_grab_frames = []  # all frames so we don't do unnecessary repeats in ffmpeg
    cue_point_grab_frames = {}  # a list specific to each cue point
    for cue_point in cue_points:
        frame_num = int((cue_point * fps) + 0.5)
        grab_frames = list(range(frame_num - 5, frame_num + 5))
        cue_point_grab_frames[str(frame_num)] = grab_frames
        all_grab_frames += grab_frames

    output_frames = ffmpeg.ffmpeg_grab_frames(
        source_path, all_grab_frames, output_dir,  214, 120, "", fps)

    # one image per cue point
    for cue_point in cue_points:
        cue_point_info = {}
        cue_point_info['time'] = cue_point
        frame_num = int((cue_point * fps) + 0.5)

        cue_point_info['frame'] = frame_num
        grab_frames = cue_point_grab_frames[str(frame_num)]
        # find these specific frames in all of the output_frames
        capture_frames = []
        for frame in grab_frames:
            frame_path = os.path.join(
                output_dir, str(frame) + ".jpg")

            if os.path.exists(frame_path):
                capture_frames.append(frame_path)

        capture_frames.insert(5, os.path.join(output_dir, "cursor.jpg"))

        output_file = os.path.join(
            output_dir, base_output_file + "_" + str(frame_num) + ".jpg")

        ffmpeg.ffmpeg_htile_images(capture_frames, output_file)

        # we want this to be a relative path
        cue_point_info['thumbnails'] = os.path.split(output_file)[1]

        results.append(cue_point_info)

    # we've got the strip, let's delete the component files
    for file in output_frames:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)

    return results


def main(package_m3u8, cue_points, output_dir):

    base_name = os.path.splitext(os.path.basename(package_m3u8))[0]
    results_dir = os.path.join(output_dir, base_name)
    os.mkdir(results_dir)

    master_m3u8 = m3u8.load(package_m3u8)

    fps = ffmpeg.ffprobe_get_framerate(package_m3u8)

    test_playlist = ""
    lowest_width = sys.maxsize

    for variant in master_m3u8.playlists:
        if variant.stream_info.resolution[1] < lowest_width:
            test_playlist = variant.absolute_uri
            lowest_width = variant.stream_info.resolution[1]

    report_data = {}
    report_data['filename'] = package_m3u8
    report_data['cue_points'] = cue_points
    report_data['cue_points_info'] = grab_thumbnails_sequence(
        test_playlist, results_dir, cue_points, fps)

    write_html_report(report_data, os.path.join(
        results_dir, base_name + ".hmtl"))


if __name__ == "__main__":
    home = expanduser("~")

    if len(sys.argv) == 4:
        package_m3u8 = sys.argv[1]
        cue_points = sys.argv[2]
        output_dir = sys.argv[3]
    else:
        # print("Usage: examine_cue_points <package_m3u8> <cue_points> <output_dir>")
        # hard coded test for now
        package_m3u8 = "https://s3-us-east-2.amazonaws.com/tubi-titan-stream-staging/ca78cdd6-9bb8-4c9e-a449-10b1486ac645/txvd3isg0e.m3u8"
        # cue_points = ffmpeg.scene_detect(package_m3u8, 0.5)
        cue_points = [7.38238, 8.46679, 14.2225, 16.5999, 25.734, 30.6556, 40.9159, 50.342, 56.9319, 58.7253, 89.9649, 90.5488,
                      92.5925, 93.3432, 94.1774, 101.143, 101.727, 103.228, 104.313, 104.98, 105.147, 106.023, 114.031, 116.825, 119.119, 119.202]
        output_dir = home + "/scene_detect"

    # create the results directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    main(package_m3u8,
         cue_points, output_dir)
