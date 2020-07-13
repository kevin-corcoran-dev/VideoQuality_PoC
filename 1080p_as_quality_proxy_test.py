import sys
import os
from os.path import expanduser
import urllib.request
import json
import subprocess
import re
import resource
import operator
from timecode import Timecode
from jinja2 import Environment, FileSystemLoader
import boto3
from botocore.exceptions import ClientError
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from fractions import Fraction
import numpy as np
from scipy.signal import find_peaks
import collections

import titan
import ffmpeg_funcs as ffmpeg
import plot_vmaf
import vmaf


tubi_video_templates = "Tubi_Templates.json"


def write_html_report(report_data, report_path):
    jinja_loader = FileSystemLoader('templates')
    jinja_env = Environment(loader=jinja_loader)
    jinja_temp = jinja_env.get_template("1080_quality_proxy.html")
    output = jinja_temp.render(data=report_data)
    print(output)
    with open(report_path, "w") as f:
        f.write(output)


def write_html_high_low_report(report_data, report_path):
    jinja_loader = FileSystemLoader('templates')
    jinja_env = Environment(loader=jinja_loader)
    jinja_temp = jinja_env.get_template("High_vs_Low_Quality.html")
    output = jinja_temp.render(data=report_data)
    print(output)
    with open(report_path, "w") as f:
        f.write(output)


def write_dict_to_json(dict, output_path):
    pretty_string = json.dumps(dict, indent=4)
    with open(output_path, 'w') as fp:
        fp.write(pretty_string)
        fp.close()


def find_first_metric_to_meet_or_exceed(compare_bitrate, metric, test_variants):
    variants_meet_or_exceed = [(variant['bitrate'], variant[metric])
                               for variant in test_variants if variant[metric] > compare_bitrate]

    if len(variants_meet_or_exceed) == 0:
        return "none", "none"
    first_pair = sorted(variants_meet_or_exceed, key=lambda x: x[0])[0]
    return first_pair[0], round(first_pair[1], 3)


def read_or_create_vmaf(source_path, transcode_path, json_path, target_fps, target_width, target_height, subsample):
    if os.path.exists(json_path):
        score, frame_nums, frame_vmafs = vmaf.parse_vmaf_json(json_path)
    else:
        score, frame_nums, frame_vmafs = vmaf.run_vmaf_full(
            source_path, transcode_path, json_path, target_fps, target_width, target_height, subsample)

    return score, frame_nums, frame_vmafs


def main(source_file, source_directory, results_directory, subsample):

    report_data = {}
    report_data['source_file'] = source_file
    report_data['comparisons'] = []
    report_data['variant_comparisons'] = collections.defaultdict(dict)
    report_data['variant_comparisons']['seven_twenty'] = {'name': "1280x720"}
    report_data['variant_comparisons']['seven_twenty']['variants'] = []
    report_data['variant_comparisons']['five_seventy_six'] = {
        'name': "1024x576"}
    report_data['variant_comparisons']['five_seventy_six'] = {}
    report_data['variant_comparisons']['five_seventy_six']['variants'] = []

    source_duration = ffmpeg.ffprobe_get_video_duration(source_file)
    report_data['source_duration'] = source_duration
    source_fps = ffmpeg.ffprobe_get_framerate(source_file)
    report_data['source_fps'] = round(source_fps, 3)

    source_frame_size = ffmpeg.ffprobe_get_frame_size(source_file)
    report_data['source_width'] = source_frame_size['width']
    report_data['source_height'] = source_frame_size['height']

    report_data['thumbnails_strip'] = ffmpeg.grab_thumbnail_strip(
        source_file, results_directory, source_duration, source_fps, 8)

    filenames = [os.path.join(source_directory, filename)
                 for filename in os.listdir(source_directory)]

    # one time through to find the largest MP4:
    largest_size = 0
    largest_file = ""
    for filename in filenames:
        if not filename.endswith(".mp4"):
            continue

        filesize = os.path.getsize(filename)
        if filesize > largest_size:
            largest_size = filesize
            largest_file = filename

    for filename in filenames:
        if not filename.endswith(".mp4"):
            continue

        basename = os.path.basename(filename)
        if basename.startswith("."):
            continue

        target_fps = ffmpeg.ffprobe_get_framerate(filename)

        compare_item = {}

        compare_item['name'] = basename

        compare_item['bitrate'] = int(re.search(
            r'(?:([0-9]*)k)', basename, re.M)[1])
        frame_size = ffmpeg.ffprobe_get_frame_size(filename)
        compare_item['width'] = frame_size['width']
        compare_item['height'] = frame_size['height']
        # vmaf vs. source
        compare_item['vs_source_json_path'] = os.path.join(
            results_directory, basename + "_vs_source_vmaf.json")
        compare_item['vs_source_diff'] = os.path.join(
            results_directory, basename + "_vs_source_diff.mp4")
        compare_item['vs_source_vmaf_plot'] = os.path.join(
            results_directory, basename + "_vs_source_vmaf.png")
        compare_item['vmaf_score_vs_source'], compare_item['vmaf_vs_source_frame_nums'], compare_item['vmaf_vs_source_frame_scores'] = read_or_create_vmaf(
            source_file, filename, compare_item['vs_source_json_path'], target_fps, compare_item['width'], compare_item['height'], subsample)

        plot_vmaf.plot_vmaf_graph(os.path.join(results_directory, compare_item['vs_source_vmaf_plot']),
                                  compare_item['vmaf_vs_source_frame_nums'], compare_item['vmaf_vs_source_frame_scores'], source_duration, None, target_fps, basename + " VMAF over duration vs. Source")

        # ffmpeg.generate_diff_movie(
        #     source_file, filename,  compare_item['vs_source_diff'], target_fps, compare_item['width'], compare_item['height'])

        # vmaf vs. 1080 high:

        compare_item['vs_1080_json_path'] = os.path.join(
            results_directory, basename + "_vs_1080_vmaf.json")
        compare_item['vs_1080_vmaf_plot'] = os.path.join(
            results_directory, basename + "_vs_1080_vmaf.png")
        compare_item['vs_1080_diff'] = os.path.join(
            results_directory, basename + "_vs_1080_diff.mp4")
        compare_item['vmaf_score_vs_1080'], compare_item['vmaf_vs_1080_frame_nums'], compare_item['vmaf_vs_1080_frame_scores'] = read_or_create_vmaf(
            largest_file, filename, compare_item['vs_1080_json_path'], target_fps, compare_item['width'], compare_item['height'], subsample)
        plot_vmaf.plot_vmaf_graph(os.path.join(results_directory, compare_item['vs_1080_vmaf_plot']),
                                  compare_item['vmaf_vs_1080_frame_nums'], compare_item['vmaf_vs_1080_frame_scores'], source_duration, None, target_fps, basename + " VMAF over duration vs. 3400kbps 1080 Reference")
        # ffmpeg.generate_diff_movie(
        #     source_file, filename,  compare_item['vs_1080_diff'], target_fps, compare_item['width'], compare_item['height'])

        # figure out the difference

        compare_item['vmaf_difference'] = [abs(source_score - score_1080) for source_score, score_1080 in zip(
            compare_item['vmaf_vs_source_frame_scores'], compare_item['vmaf_vs_1080_frame_scores'])]

        compare_item['vmaf_difference_plot'] = os.path.join(
            results_directory, basename + "_vmaf_difference.png")

        plot_vmaf.plot_vmaf_graph(os.path.join(results_directory, compare_item['vmaf_difference_plot']),
                                  compare_item['vmaf_vs_1080_frame_nums'], compare_item['vmaf_difference'], source_duration, None, target_fps, "Absolute differnce between VMAF vs. Source and VMAF vs. Reference")

        compare_item['average_vmaf_difference'] = round(sum(
            compare_item['vmaf_difference']) / len(compare_item['vmaf_difference']), 3)

        compare_item['max_vmaf_difference'] = round(max(
            compare_item['vmaf_difference']), 3)

        report_data['comparisons'].append(compare_item)

        if compare_item['height'] == 720:
            report_data['variant_comparisons']['seven_twenty']['variants'].append(
                compare_item)
        if compare_item['height'] == 576:
            report_data['variant_comparisons']['five_seventy_six']['variants'].append(
                compare_item)

    # compare the matching variants
    for variant_comparison in report_data['variant_comparisons'].items():
        variant_comparison[1]['vmaf_difference'] = [abs(low - high) for low, high in zip(
            variant_comparison[1]['variants'][0]['vmaf_vs_1080_frame_scores'], variant_comparison[1]['variants'][1]['vmaf_vs_1080_frame_scores'])]

        variant_comparison[1]['average_vmaf_difference'] = round(sum(
            variant_comparison[1]['vmaf_difference']) / len(variant_comparison[1]['vmaf_difference']), 3)

        variant_comparison[1]['max_vmaf_difference'] = round(max(
            variant_comparison[1]['vmaf_difference']), 3)

        variant_comparison[1]['vmaf_difference_plot'] = os.path.join(
            results_directory,  str(variant_comparison[1]['variants'][0]['height']) + "_" + str(variant_comparison[1]['variants'][0]['bitrate']) + "_vs_" + str(variant_comparison[1]['variants'][1]['bitrate']) + ".png")

        plot_vmaf.plot_vmaf_graph(variant_comparison[1]['vmaf_difference_plot'],
                                  variant_comparison[1]['variants'][0]['vmaf_vs_1080_frame_nums'], variant_comparison[1]['vmaf_difference'], source_duration, None, target_fps, "Absolute differnce High vs. Low bitrate VMAF vs. 1080")

    # end compare variants

    report_data['comparisons'] = sorted(report_data['comparisons'],
                                        key=lambda k: (k['width'], k['bitrate']), reverse=True)

    write_dict_to_json(report_data, os.path.join(
        results_directory, basename + "_1080p_as_quality_proxy.json"))

    write_html_report(report_data, os.path.join(
        results_directory, "VMAFvsSourcevsVMAFvs1080.html"))

    write_html_high_low_report(report_data, os.path.join(
        results_directory, "HighVsLow.html"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        source_directory = sys.argv[1]
        results_directory = sys.argv[2]
        subsample = sys.argv[3]
    else:
        home = expanduser('~')
        # hardcoded for debugging
        # in the future return an error instead

        # source_directory = home + "/Downloads/Lawless_1min_bitrate_conservation_test"
        # source_file = os.path.join(
        #     source_directory, "LawlessKingdom_1min.mov")

        source_directory = home + "/Downloads/FairyTail_bitrate_conservation_test"
        source_file = os.path.join(
            source_directory, "FunimationVenue_FairyTail_1min.mov")

        results_directory = source_directory + "/results/"
        subsample = 1

    # create the results directory
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    # sys.argv[1], sys.argv[2]
    main(source_file, source_directory,
         results_directory, subsample)
