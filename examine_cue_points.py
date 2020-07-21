import sys
import os
import m3u8
from os.path import expanduser

import numpy as np
import ffmpeg_funcs as ffmpeg


def grab_thumbnails_sequence(source_path, output_dir, frame_nums, fps):

    if not os.path.isdir(output_dir):
        raise Exception(
            'output_directory must be a directory: {}'.format(output_dir))

    output_file = os.path.basename(source_path)
    output_file = os.path.splitext(output_file)[0]
    output_file = os.path.join(output_dir, output_file + "_thumbnails.jpg")

    output_frames = ffmpeg.ffmpeg_grab_frames(
        source_path, frame_nums, output_dir,  214, 120, "frame_", fps)

    # ffmpeg_funcs.ffmpeg_grab_frames() returns relative paths -- let's turn them into absolute
    output_frames = [os.path.join(output_dir, output_frame)
                     for output_frame in output_frames]

    ffmpeg.ffmpeg_htile_images(output_frames, output_file)

    # we've got the strip, let's delete the component files
    for file in output_frames:
        os.remove(file)

    return output_file


def main(package_m3u8, cue_points, output_dir):
    master_m3u8 = m3u8.load(package_m3u8)

    fps = ffmpeg.ffprobe_get_framerate(package_m3u8)

    test_playlist = ""
    lowest_width = sys.maxsize

    for variant in master_m3u8.playlists:
        if variant.stream_info.resolution[1] < lowest_width:
            test_playlist = variant.absolute_uri
            lowest_width = variant.stream_info.resolution[1]

    for cue_point in cue_points:
        frame_num = int(cue_point * fps)
        grabframes = list(range(frame_num - 5, frame_num + 5))
        grab_thumbnails_sequence(
            test_playlist, output_dir, grabframes, fps)


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
