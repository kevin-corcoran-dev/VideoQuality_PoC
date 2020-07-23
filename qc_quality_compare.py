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
from fractions import Fraction
import numpy as np

import titan
import ffmpeg_funcs as ffmpeg
import vmaf
import plot_vmaf
import ffargs


tubi_video_templates = "Tubi_Templates.json"


def write_html_report(report_data, report_path):
    jinja_loader = FileSystemLoader('templates')
    jinja_env = Environment(loader=jinja_loader)
    jinja_temp = jinja_env.get_template("crf_comparison.html")
    output = jinja_temp.render(data=report_data)
    print(output)
    with open(report_path, "w") as f:
        f.write(output)


def write_dict_to_json(dict, output_path):
    pretty_string = json.dumps(dict, indent=4)
    with open(output_path, 'w') as fp:
        fp.write(pretty_string)
        fp.close()


def write_json_entry(output_path, key, value):
    entries = {}

    if os.path.exists(output_path):
        with open(output_path, 'r') as fp:
            entries = json.load(fp)
            fp.close()

    entries[key] = value

    with open(output_path, 'w') as fp:
        pretty_string = json.dumps(entries, indent=4)
        fp.write(pretty_string)
        fp.close()


def read_json_entry(output_path, key):
    entries = {}

    if os.path.exists(output_path):
        with open(output_path, 'r') as fp:
            entries = json.load(fp)

            if key in entries:
                return entries[key]
            else:
                return None
    else:
        return None


def transcode_new_or_used(source_path, template, output_path, results_dict):

    if not os.path.exists(output_path):
        results_dict['transcode_time'] = ffmpeg.transcode(
            source_path, template, output_path)
        write_json_entry(results_dict['transcode_times_json'],
                         output_path, results_dict['transcode_time'])
    elif 'transcode_times_json' in results_dict and os.path.exists(results_dict['transcode_times_json']):
        results_dict['transcode_time'] = read_json_entry(
            results_dict['transcode_times_json'], output_path)
    else:
        results_dict['transcode_time'] = 0
        write_json_entry(results_dict['transcode_times_json'],
                         os.path.split(output_path)[1], 0)

    results_dict['file_size'] = os.path.getsize(output_path)


def create_x_vs_y_plot(data, x_axis, y_axis, y_range, out_dir, base_file_name):
    if not "plots" in data:
        data['plots'] = {}

    template_name = data['template_name']

    plot_key = x_axis + "_vs_" + y_axis
    data['plots'][plot_key] = os.path.join(out_dir, base_file_name + "_" + template_name +
                                           "_" + plot_key + ".png")

    plot_vmaf.plot_vmaf_vs_something(data['plots'][plot_key],  template_name, x_axis, data[y_axis],
                                     data[x_axis], y_range, y_axis, data['test_variants'])


def main(source_directory, results_directory, subsample):

    report_data = {}
    report_data['title'] = "Bitrates"

    with open('tubi_video_templates.json') as f:
        templates = json.load(f)

    report_data['test_assets'] = []
    for filename in os.listdir(source_directory):

        # if not filename == "Bloomberg.mp4":
        #     continue

        if not filename.endswith(".mov") and not filename.endswith(".mp4"):
            continue

        abs_source_path = os.path.join(source_directory, filename)

        asset_report = {}
        asset_report['filepath'] = filename

        asset_report['fps'] = fps = ffmpeg.ffprobe_get_framerate(
            abs_source_path)
        source_dimensions = ffmpeg.ffprobe_get_frame_size(abs_source_path)
        asset_report['source_width'] = source_dimensions['width']
        asset_report['source_height'] = source_dimensions['height']
        asset_report['duration'] = duration = ffmpeg.ffprobe_get_video_duration(
            abs_source_path)
        asset_report['deinterlace'] = deinterlace = ffmpeg.ffmpeg_get_deinterlace(
            abs_source_path, duration)
        asset_report['SAR'] = sample_aspect_ratio = ffmpeg.ffprobe_get_sample_aspect_ratio(
            abs_source_path)

        base_file_name = os.path.splitext(os.path.split(filename)[1])[0]

        out_dir = os.path.join(results_directory, base_file_name)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        # grab the thumbnails
        asset_report['thumbnails_strip'] = ffmpeg.grab_thumbnail_strip(
            abs_source_path, out_dir, duration, fps, 7)

        asset_report['variants'] = []
        for template in templates:

            target_width = template['target_width']
            target_height = template['target_height']
            target_fps = titan.generate_fps_float(fps)

            # no uprezzing
            if target_width > source_dimensions['width'] and target_height > source_dimensions['height']:
                continue

            template_results = {}
            template_results['template_name'] = template_name = template['description']
            template_results['template'] = template
            template_results['transcode_times_json'] = os.path.join(
                out_dir, base_file_name + "_transcode_times.json")

            output_path = os.path.join(
                out_dir, base_file_name + "_" + template_name + ".mp4")
            json_path = os.path.join(
                out_dir, base_file_name + "_" + template_name + ".json")

            base_template = template['template']
            template_results['crf'] = template['crf'] = ffargs.get_crf(
                base_template)

            key_frame_placement = titan.generate_key_frame_placement(
                target_fps)
            video_filter = titan.generate_video_filter(
                target_width, target_height, fps, sample_aspect_ratio, deinterlace, source_dimensions['width'], source_dimensions['height'])

            vod_template = titan.customize_template(
                base_template, video_filter, key_frame_placement)

            transcode_new_or_used(
                abs_source_path, vod_template, output_path, template_results)

            # if not os.path.exists(output_path):
            #     template_results['transcode_time'] = ffmpeg.transcode(
            #         abs_source_path, vod_template, output_path)
            #     write_json_entry(asset_report['transcode_times_json'],
            #                      output_path, template_results['transcode_time'])
            # else:
            #     template_results['transcode_time'] = read_json_entry(
            #         asset_report['transcode_times_json'], template_results['transcode_time'])

            # template_results['file_size'] = os.path.getsize(output_path)

            # save time if we have this file from a previous run -- for debugging
            score, frame_nums, frame_vmafs = vmaf.read_or_create_vmaf(
                abs_source_path, output_path, json_path, target_fps, target_width, target_height, subsample)

            template_results['vmaf'] = round(score, 3)
            template_results['vod_vmaf_frames'] = frame_nums
            template_results['vod_vmaf_frame_scores'] = frame_vmafs
            template_results['min_vmaf'] = min(frame_vmafs)
            template_results['max_vmaf'] = max(frame_vmafs)

            # same thing now for SSIM
            ssim, _, frame_ssim_scores = vmaf.parse_ssim_from_vmaf_json(
                json_path)
            template_results['ssim'] = round(ssim, 3)
            template_results['vod_ssim_frame_scores'] = frame_ssim_scores

            # same thing now for PSNR
            psnr, _, frame_psnr_scores = vmaf.parse_psnr_from_vmaf_json(
                json_path)
            template_results['psnr'] = round(psnr, 3)
            template_results['vod_psnr_frame_scores'] = frame_psnr_scores

            alternate_crf_targets = ffargs.generate_crf_range(1, 30, 15)
            alternate_crf_targets[-1] = 1

            template_results['test_variants'] = []
            for alternate_crf in alternate_crf_targets:
                test_variant = {}
                test_variant['transcode_times_json'] = template_results['transcode_times_json']

                test_variant['crf'] = alternate_crf
                test_variant['output_path'] = os.path.join(
                    out_dir, base_file_name + "_" + template_name + "_crf_" + str(alternate_crf) + ".mp4")
                test_variant['vmaf_json_path'] = os.path.join(
                    out_dir, base_file_name + "_" + template_name + "_crf_" + str(alternate_crf) + "_vmaf.json")

                test_template = ffargs.change_crf(
                    vod_template, alternate_crf)

                transcode_new_or_used(
                    abs_source_path, test_template, test_variant['output_path'], test_variant)

                # if not os.path.exists(test_variant['output_path']):
                #     test_variant['transcode_time'] = ffmpeg.transcode(
                #         abs_source_path, test_template, test_variant['output_path'])
                # else:
                #     test_variant['transcode_time'] = 0

                # test_variant['file_size'] = os.path.getsize(
                #     test_variant['output_path'])

                test_variant['vmaf'], test_variant['vmaf_frame_times'], test_variant['vmaf_frame_scores'] = vmaf.read_or_create_vmaf(
                    abs_source_path, test_variant['output_path'], test_variant['vmaf_json_path'], target_fps, target_width, target_height, subsample)

                test_variant['ssim'], _, test_variant['ssim_frame_scores'] = vmaf.parse_ssim_from_vmaf_json(
                    test_variant['vmaf_json_path'])

                test_variant['psnr'], _, test_variant['psnr_frame_scores'] = vmaf.parse_psnr_from_vmaf_json(
                    test_variant['vmaf_json_path'])

                test_variant['min_vmaf'] = min(
                    test_variant['vmaf_frame_scores'])
                test_variant['max_vmaf'] = max(
                    test_variant['vmaf_frame_scores'])

                template_results['test_variants'].append(test_variant)

            for y_axis in ['vmaf', 'ssim', 'psnr', 'file_size', 'transcode_time', 'min_vmaf', 'max_vmaf']:
                create_x_vs_y_plot(template_results, y_axis, 'crf', alternate_crf_targets,
                                   out_dir, base_file_name)

            template_results['multi_crf_time_plot'] = os.path.join(out_dir, base_file_name + "_" + template_name +
                                                                   "_multi_VMAF_vs_time.png")

            plot_vmaf.plot_multi_vmaf_timegraph(template_results['multi_crf_time_plot'], template_results['vod_vmaf_frames'],
                                                template_results['vod_vmaf_frame_scores'], template_results['test_variants'], template['crf'], duration, asset_report['fps'], 'crf')

            asset_report['variants'].append(template_results)

        report_data['test_assets'].append(asset_report)

    write_dict_to_json(report_data, os.path.join(
        results_directory, base_file_name + "_bitrate_tests.json"))

    write_html_report(report_data, os.path.join(
        results_directory, "crf_comparison.html"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        source_directory = sys.argv[1]
        results_directory = sys.argv[2]
        subsample = sys.argv[3]
    else:
        # hardcoded for debugging
        # in the future return an error instead
        source_directory = expanduser(
            '~') + "/Downloads/golden_reference_research/"
        results_directory = expanduser(
            '~') + "/video_quality/CQ_testing"
        subsample = 1

    # create the results directory
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    # sys.argv[1], sys.argv[2]
    main(source_directory,
         results_directory, subsample)
