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
from fractions import Fraction
import numpy as np
from scipy.signal import find_peaks

import titan
import ffmpeg_funcs


tubi_video_templates = "Tubi_Templates.json"


def ffmpeg_grab_thumbnail_strip(source_path, output_dir, duration, fps, steps):

    if not os.path.isdir(results_directory):
        raise Exception(
            'output_directory must be a directory: {}'.format(results_directory))

    output_file = os.path.basename(source_path)
    output_file = os.path.splitext(output_file)[0]

    output_file = os.path.join(output_dir, output_file + "_thumbnails.jpg")

    times = np.linspace(0, duration - 0.1, steps)
    frame_nums = [int(time * fps) for time in times]

    output_frames = ffmpeg_funcs.ffmpeg_grab_frames(
        source_path, frame_nums, output_dir,  214, 120, "frame_", fps)

    # ffmpeg_funcs.ffmpeg_grab_frames() returns relative paths -- let's turn them into absolut
    output_frames = [os.path.join(output_dir, output_frame)
                     for output_frame in output_frames]

    ffmpeg_funcs.ffmpeg_htile_images(output_frames, output_file)

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


def ffmpeg_run_vmaf(source, transcode, ss, t, dest_fps, dest_size):

    args = ["ffmpeg", "-ss", str(ss), "-i", source, "-t", str(t),
            "-ss", str(ss), "-i", transcode, "-t", str(t), "-an",
            "-filter_complex", "[0:v]fps=" + str(dest_fps) + ",scale=" + str(dest_size['width']) + ":" + str(dest_size['height']) + ",setpts=PTS-STARTPTS[source];[1:v]setpts=PTS-STARTPTS[trancoded];[source][trancoded]libvmaf", "-f", "NULL", " -"]

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    separator = " "
    return parse_vmaf(result.stderr), separator.join(args)


def ffmpeg_run_vmaf_full(source, transcode, log_path, dest_fps, target_width, target_height, subsample=1):

    if not log_path.endswith(".json"):
        raise Exception('log_path must end in .json: {}'.format(log_path))

    args = ["ffmpeg", "-i", source, "-i", transcode, "-an",
            "-filter_complex", "[0:v]fps=" + str(dest_fps) + ",scale=" + str(target_width) + ":" + str(target_height) + ",setpts=PTS-STARTPTS[source];[1:v]setpts=PTS-STARTPTS[trancoded];[source][trancoded]libvmaf=log_fmt=json:psnr=1:ssim=1:log_path=" + log_path + ":n_subsample=" + str(subsample), "-f", "NULL", " -"]

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    result.check_returncode()

    return parse_vmaf_json(log_path)


def parse_vmaf(ffmepgstderr):
    matched_lines = [line for line in ffmepgstderr.split(
        '\n') if "libvmaf" in line and "VMAF score:" in line]

    vmaf_match = re.search(r'(?:VMAF score: )([0-9.]*)', matched_lines[0])

    fVmaf = float(vmaf_match.group(1))

    return fVmaf


def parse_vmaf_json(json_path):

    vmaf_frames = []
    vmav_frame_times = []
    vmaf_score = 0

    with open(json_path, 'r') as f:
        vmaf_json = json.load(f)

        vmaf_score = vmaf_json['VMAF score']

        for frame in vmaf_json['frames']:
            vmav_frame_times.append(frame['frameNum'])
            vmaf_frames.append(frame['metrics']['vmaf'])

        return vmaf_score, vmav_frame_times, vmaf_frames


def parse_psnr_from_vmaf_json(json_path):

    psnr_frame_scores = []
    psnr_frame_nums = []
    psnr_score = 0

    with open(json_path, 'r') as f:
        vmaf_json = json.load(f)

        psnr_score = vmaf_json['PSNR score']

        for frame in vmaf_json['frames']:
            psnr_frame_nums.append(frame['frameNum'])
            psnr_frame_scores.append(frame['metrics']['psnr'])

        return psnr_score, psnr_frame_nums, psnr_frame_scores


def parse_ssim_from_vmaf_json(json_path):

    ssim_frame_scores = []
    ssim_frame_nums = []
    ssim_score = 0

    with open(json_path, 'r') as f:
        vmaf_json = json.load(f)

        ssim_score = vmaf_json['SSIM score']

        for frame in vmaf_json['frames']:
            ssim_frame_nums.append(frame['frameNum'])
            ssim_frame_scores.append(frame['metrics']['ssim'])

        return ssim_score, ssim_frame_nums, ssim_frame_scores


def plot_vmaf_graph(output_path, frame_nums, frame_scores, source_duration, lowest_values, timecode):
    plt.plot(frame_nums, frame_scores)

    # generate major tics based on evenly divided time

    times = np.linspace(0, source_duration, 20)

    # ticframes = [int(float(timecode.framerate) * float(time))
    #              for time in times]

    # ticlabels = [str(*timecode.float_to_tc(time)) for time in times]
    ticframes = [timecode.float_to_tc(float(scene))
                 for scene in times]
    ticlabels = [timecode.tc_to_string(*timecode.frames_to_tc(ticframe))
                 for ticframe in ticframes]
    # double back and modify tick frames by subsample
    ticframes = [frame for frame in ticframes]

    plt.xticks(ticframes, ticlabels, rotation='vertical')

    # only about 25 labels are going to fit comfortably, so we need to
    # turn off excess labels
    max_labels = 25
    label_mod = int(len(ticlabels)/max_labels)

    if label_mod == 0:
        label_mod = 1

    ax = plt.axes()
    style = dict(size=10, color='gray')
    # label the valleys
    for idx, lowval in enumerate(lowest_values):
        ax.text(lowval, frame_scores[lowval] - 5, str(idx + 1), **style)

    ax.set_xticks(ticframes, minor=True)

    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % label_mod != 0:
            label.set_visible(False)

    ax.grid()

    plt.ylabel('vmaf score')
    plt.ylim(0, 100)
    plt.xlabel('time')
    plt.subplots_adjust(bottom=0.3)
    plt.gcf().set_size_inches(15, 5)
    plt.savefig(output_path)


def plot_vmaf_vs_bitrate(output_path, metric_name, reference_bitrate, reference_vmaf, bitrates, variants):

    bitrate_vmafs = [variant[metric_name] for variant in variants]

    if metric_name == 'ssim':
        lower = min(bitrate_vmafs) - 0.01
        upper = 0.01 + max(bitrate_vmafs)
    else:
        lower = round(min(bitrate_vmafs) - 1)
        upper = round(1 + max(bitrate_vmafs))

    # round our range to even multiples of 5
    # base = 5
    # lower = base * round(lower/base)
    # upper = base * (1 + round(upper/base))

    title = "Bitrate vs. " + metric_name + " for 2-second GOPS"
    plt.title(title, fontsize=14, color='blue')
    plt.plot(bitrates, bitrate_vmafs)
    # plt.plot([reference_bitrate, reference_bitrate, 0], [
    #         0, reference_vmaf, 100], lw=1, dashes=[2, 2])
    plt.axvline(reference_bitrate, color='r')
    plt.axhline(reference_vmaf, color='r')

    ticlabels = [str(bitrate) for bitrate in bitrates]
    plt.xticks(bitrates, ticlabels, rotation='vertical')

    ax = plt.axes()
    # style = dict(size=10, color='gray')
    # # label the valleys
    # for idx, lowval in enumerate(lowest_values):
    #     ax.text(lowval, frame_scores[lowval] - 5, str(idx + 1), **style)

    # ax.set_xticks(ticframes, minor=True)

    # for index, label in enumerate(ax.xaxis.get_ticklabels()):
    #     if index % label_mod != 0:
    #         label.set_visible(False)

    ax.grid()

    plt.ylabel(metric_name + " score")
    plt.ylim(lower, upper)
    plt.xlabel('bitrate')
    plt.subplots_adjust(bottom=0.2)
    plt.gcf().set_size_inches(7, 5)
    plt.savefig(output_path)
    plt.clf()


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
    # return list(range(original_target - 200000, original_target + 600000, 100000))
    new_targets = np.linspace(int(original_target/2),
                              int(original_target * 2), 10)
    return [int(target) for target in new_targets]


def ffmpeg_transcode(source_path, video_template, output_path):

    video_template = video_template.split()
    args = ["ffmpeg", "-y", "-i", source_path, ] + \
        video_template + [output_path]

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    if result.returncode:
        print(result.stderr)

    result.check_returncode()


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

        asset_report['fps'] = fps = ffprobe_get_framerate(abs_source_path)
        source_dimensions = ffprobe_get_frame_size(abs_source_path)
        asset_report['source_width'] = source_dimensions['width']
        asset_report['source_height'] = source_dimensions['height']
        asset_report['duration'] = duration = ffprobe_get_video_duration(
            abs_source_path)
        asset_report['deinterlace'] = deinterlace = ffmpeg_get_deinterlace(
            abs_source_path, duration)
        asset_report['SAR'] = sample_aspect_ratio = ffprobe_get_sample_aspect_ratio(
            abs_source_path)

        base_file_name = os.path.splitext(os.path.split(filename)[1])[0]

        out_dir = os.path.join(results_directory, base_file_name)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        # grab the thumbnails
        asset_report['thumbnails_strip'] = ffmpeg_grab_thumbnail_strip(
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

            image_path = base_file_name + "vmaf.png"

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
                ffmpeg_transcode(
                    abs_source_path, vod_template, output_path)

            # save time if we have this file from a previous run -- for debugging
            if os.path.exists(json_path):
                score, frame_nums, frame_vmafs = parse_vmaf_json(json_path)
            else:
                score, frame_nums, frame_vmafs = ffmpeg_run_vmaf_full(
                    abs_source_path, output_path, json_path, target_fps, target_width, target_height, subsample)

            template_results['vod_vmaf_score'] = round(score, 3)
            template_results['vod_vmaf_frames'] = frame_nums
            template_results['vod_vmaf_frame_scores'] = frame_vmafs

            # same thing now for SSIM
            ssim, _, frame_ssim_scores = parse_ssim_from_vmaf_json(json_path)
            template_results['vod_ssim_score'] = round(ssim, 3)
            template_results['vod_ssim_frame_scores'] = frame_ssim_scores

            # same thing now for PSNR
            psnr, _, frame_psnr_scores = parse_psnr_from_vmaf_json(json_path)
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
                    ffmpeg_transcode(
                        abs_source_path, test_template, test_variant['output_path'])

                if os.path.exists(test_variant['vmaf_json_path']):
                    test_variant['vmaf'], test_variant['vmaf_frame_times'], test_variant['vmaf_frame_scores'] = parse_vmaf_json(
                        test_variant['vmaf_json_path'])
                else:
                    test_variant['vmaf'], test_variant['vmaf_frame_times'], test_variant['vmaf_frame_scores'] = ffmpeg_run_vmaf_full(
                        abs_source_path, test_variant['output_path'], test_variant['vmaf_json_path'], target_fps, target_width, target_height, subsample)

                test_variant['ssim'], _, test_variant['ssim_frame_scores'] = parse_ssim_from_vmaf_json(
                    test_variant['vmaf_json_path'])

                test_variant['psnr'], _, test_variant['psnr_frame_scores'] = parse_psnr_from_vmaf_json(
                    test_variant['vmaf_json_path'])

                template_results['test_variants'].append(test_variant)

            template_results['bitrate_vs_vmaf_plot'] = os.path.join(out_dir, base_file_name + "_" + template_name +
                                                                    "_vmaf_vs_bitrate.png")

            template_results['bitrate_vs_ssim_plot'] = os.path.join(out_dir, base_file_name + "_" + template_name +
                                                                    "_ssim_vs_bitrate.png")

            plot_vmaf_vs_bitrate(template_results['bitrate_vs_vmaf_plot'], "vmaf", template['target_bitrate'],
                                 template_results['vod_vmaf_score'], alternate_bitrate_targets, template_results['test_variants'])

            plot_vmaf_vs_bitrate(template_results['bitrate_vs_ssim_plot'], "ssim", template['target_bitrate'],
                                 template_results['vod_ssim_score'], alternate_bitrate_targets, template_results['test_variants'])

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
        source_directory = expanduser('~') + "/Downloads/video_quality/"
        results_directory = expanduser('~') + "/video_quality/live_news"
        subsample = 1

    # create the results directory
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    # sys.argv[1], sys.argv[2]
    create_quality_report(source_directory,
                          results_directory, subsample)
