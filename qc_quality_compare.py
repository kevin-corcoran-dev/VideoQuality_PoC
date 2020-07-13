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


tubi_video_templates = "Tubi_Templates.json"


def write_html_report(report_data, report_path):
    jinja_loader = FileSystemLoader('templates')
    jinja_env = Environment(loader=jinja_loader)
    jinja_temp = jinja_env.get_template("bitrate_comparison.html")
    output = jinja_temp.render(data=report_data)
    print(output)
    with open(report_path, "w") as f:
        f.write(output)


def write_dict_to_json(dict, output_path):
    pretty_string = json.dumps(dict, indent=4)
    with open(output_path, 'w') as fp:
        fp.write(pretty_string)
        fp.close()


def generate_alternate_bitrates(original_target):
    new_targets = np.linspace(int(original_target/2),
                              int(original_target * 2), 10)
    return [int(target) for target in new_targets]


def generate_qc_factors(proposed_target):
    new_targets = np.linspace(int(proposed_target/2),
                              int(proposed_target * 2), 10)
    return [int(target) for target in new_targets]


def find_first_metric_to_meet_or_exceed(compare_bitrate, metric, test_variants):
    variants_meet_or_exceed = [(variant['bitrate'], variant[metric])
                               for variant in test_variants if variant[metric] > compare_bitrate]

    if len(variants_meet_or_exceed) == 0:
        return "none", "none"
    first_pair = sorted(variants_meet_or_exceed, key=lambda x: x[0])[0]
    return first_pair[0], round(first_pair[1], 3)


def create_quality_report(source_directory, results_directory, subsample):

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
            abs_source_path, out_dir, duration, fps, 5)

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
            output_path = os.path.join(
                out_dir, base_file_name + "_" + template_name + ".mp4")
            json_path = os.path.join(
                out_dir, base_file_name + "_" + template_name + ".json")

            base_template = template['template']

            key_frame_placement = titan.generate_key_frame_placement(
                target_fps)
            video_filter = titan.generate_video_filter(
                target_width, target_height, fps, sample_aspect_ratio, deinterlace, source_dimensions['width'], source_dimensions['height'])

            base_template = base_template.replace(
                "{{ video_filter }}", video_filter)

            vod_template = base_template.replace(
                "{{ key_frame_placement }}", key_frame_placement)

            linear_template = base_template.replace(
                "{{ key_frame_placement }}", titan.generate_custom_key_frame_placement(target_fps, 2))

            if not os.path.exists(output_path):
                ffmpeg.transcode(
                    abs_source_path, vod_template, output_path)

            # save time if we have this file from a previous run -- for debugging
            if os.path.exists(json_path):
                score, frame_nums, frame_vmafs = vmaf.parse_vmaf_json(
                    json_path)
            else:
                score, frame_nums, frame_vmafs = vmaf.run_vmaf_full(
                    abs_source_path, output_path, json_path, target_fps, target_width, target_height, subsample)

            template_results['vod_vmaf_score'] = round(score, 3)
            template_results['vod_vmaf_frames'] = frame_nums
            template_results['vod_vmaf_frame_scores'] = frame_vmafs

            # same thing now for SSIM
            ssim, _, frame_ssim_scores = vmaf.parse_ssim_from_vmaf_json(
                json_path)
            template_results['vod_ssim_score'] = round(ssim, 3)
            template_results['vod_ssim_frame_scores'] = frame_ssim_scores

            # same thing now for PSNR
            psnr, _, frame_psnr_scores = vmaf.parse_psnr_from_vmaf_json(
                json_path)
            template_results['vod_psnr_score'] = round(psnr, 3)
            template_results['vod_psnr_frame_scores'] = frame_psnr_scores

            alternate_bitrate_targets = generate_alternate_bitrates(
                template['target_bitrate'])

            template_results['test_variants'] = []
            for alternate_bitrate_target in alternate_bitrate_targets:
                test_variant = {}

                test_variant['bitrate'] = alternate_bitrate_target
                test_variant['output_path'] = os.path.join(
                    out_dir, base_file_name + "_" + template_name + "_" + str(alternate_bitrate_target) + ".mp4")
                test_variant['vmaf_json_path'] = os.path.join(
                    out_dir, base_file_name + "_" + template_name + "_" + str(alternate_bitrate_target) + "_vmaf.json")

                test_template = linear_template
                altnernate_bitrate_target_k = str(
                    int(alternate_bitrate_target/1000)) + "k"
                original_bitrate_target_k = str(
                    int(template['target_bitrate']/1000)) + "k"

                test_template = test_template.replace(
                    original_bitrate_target_k, altnernate_bitrate_target_k)

                original_buffersize = re.search(
                    r'(?:bufsize )([0-9.]*k)', test_template, re.S)[0]
                alternate_buffer_target = "bufsize " + str(
                    int(0.8 * alternate_bitrate_target/1000)) + "k"

                test_template = test_template.replace(
                    original_buffersize, alternate_buffer_target)

                print(test_template)

                if not os.path.exists(test_variant['output_path']):
                    ffmpeg.transcode(
                        abs_source_path, test_template, test_variant['output_path'])

                if os.path.exists(test_variant['vmaf_json_path']):
                    test_variant['vmaf'], test_variant['vmaf_frame_times'], test_variant['vmaf_frame_scores'] = vmaf.parse_vmaf_json(
                        test_variant['vmaf_json_path'])
                else:
                    test_variant['vmaf'], test_variant['vmaf_frame_times'], test_variant['vmaf_frame_scores'] = vmaf.run_vmaf_full(
                        abs_source_path, test_variant['output_path'], test_variant['vmaf_json_path'], target_fps, target_width, target_height, subsample)

                test_variant['ssim'], _, test_variant['ssim_frame_scores'] = vmaf.parse_ssim_from_vmaf_json(
                    test_variant['vmaf_json_path'])

                test_variant['psnr'], _, test_variant['psnr_frame_scores'] = vmaf.parse_psnr_from_vmaf_json(
                    test_variant['vmaf_json_path'])

                template_results['test_variants'].append(test_variant)

            # find the lowest VMAF, SSIM, and PSNR to match VMAF/SSIM/PSNR of VOD transcode
            template_results['first_acceptable_vmaf_bitrate'], template_results['first_acceptable_vmaf'] = find_first_metric_to_meet_or_exceed(
                template_results['vod_vmaf_score'], "vmaf", template_results['test_variants'])
            template_results['first_acceptable_ssim_bitrate'], template_results['first_acceptable_ssim'] = find_first_metric_to_meet_or_exceed(
                template_results['vod_ssim_score'], "ssim", template_results['test_variants'])
            template_results['first_acceptable_psnr_bitrate'], template_results['first_acceptable_psnr'] = find_first_metric_to_meet_or_exceed(
                template_results['vod_psnr_score'], "psnr", template_results['test_variants'])

            # get the name for our charts
            template_results['bitrate_vs_vmaf_plot'] = os.path.join(out_dir, base_file_name + "_" + template_name +
                                                                    "_vmaf_vs_bitrate.png")
            template_results['bitrate_vs_ssim_plot'] = os.path.join(out_dir, base_file_name + "_" + template_name +
                                                                    "_ssim_vs_bitrate.png")

            template_results['multi_bitrate_time_plot'] = os.path.join(out_dir, base_file_name + "_" + template_name +
                                                                       "_multi_VMAF_vs_time.png")
            template_results['three_bitrate_time_plot'] = os.path.join(out_dir, base_file_name + "_" + template_name +
                                                                       "_triple_VMAF_vs_time.png")

            # Generate the charts
            plot_vmaf.plot_vmaf_vs_bitrate(template_results['bitrate_vs_vmaf_plot'],  template_name, "vmaf", template['target_bitrate'],
                                           template_results['vod_vmaf_score'], alternate_bitrate_targets, template_results['test_variants'])
            plot_vmaf.plot_vmaf_vs_bitrate(template_results['bitrate_vs_ssim_plot'], template_name, "ssim", template['target_bitrate'],
                                           template_results['vod_ssim_score'], alternate_bitrate_targets, template_results['test_variants'])
            plot_vmaf.plot_multi_vmaf_timegraph(template_results['multi_bitrate_time_plot'], template_results['vod_vmaf_frames'],
                                                template_results['vod_vmaf_frame_scores'], template_results['test_variants'], template['target_bitrate'], duration, asset_report['fps'])
            plot_vmaf.plot_three_vmaf_timegraph(template_results['three_bitrate_time_plot'], template_results['vod_vmaf_frames'],
                                                template_results['vod_vmaf_frame_scores'], template_results['test_variants'], template['target_bitrate'], template_results['first_acceptable_vmaf_bitrate'], duration, asset_report['fps'])

            asset_report['variants'].append(template_results)

        report_data['test_assets'].append(asset_report)

        # report_data['vmaf_score'] = score
        # report_data['vmaf_graph'] = image_path
        # duration_tc = timecode.frames_to_tc(
        #     timecode.float_to_tc(source_video_duration))
        # report_data['duration'] = timecode.tc_to_string(*duration_tc)
        # report_data['full_vmaf_json'] = json_path
        # report_data['low_frames'] = low_frames

        # # write the unformatted json version of our results
        # write_dict_to_json(report_data, os.path.join(
        #     results_directory, media_id + "_QC.json"))
        # # write the formatted html version of our results

    write_dict_to_json(report_data, os.path.join(
        results_directory, base_file_name + "_bitrate_tests.json"))

    write_html_report(report_data, os.path.join(
        results_directory, "bitrate_comparison.html"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        source_directory = sys.argv[1]
        results_directory = sys.argv[2]
        subsample = sys.argv[3]
    else:
        # hardcoded for debugging
        # in the future return an error instead
        # source_directory = expanduser('~') + "/Downloads/video_quality/"
        # results_directory = expanduser('~') + "/video_quality/live_news"
        # subsample = 1

        source_directory = expanduser(
            '~') + "/Downloads/golden_reference_research/"
        results_directory = expanduser(
            '~') + "/video_quality/golden_reference_research"
        subsample = 1

    # create the results directory
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    # sys.argv[1], sys.argv[2]
    create_quality_report(source_directory,
                          results_directory, subsample)
