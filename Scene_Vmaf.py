import sys
import os
import paramiko
import psycopg2
from paramiko import SSHClient
from sshtunnel import SSHTunnelForwarder
from os.path import expanduser
import urllib.request
import m3u8
import json
import subprocess
import re
import s3fs
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


def split_s3_path(s3_path):
    path_parts = s3_path.replace("s3://", "").split("/")
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)
    filename = path_parts[-1]
    return bucket, key, filename


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


def ffprobe_get_frame_size(source):
    result = subprocess.run(["ffprobe", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "default=noprint_wrappers=1", source],
                            shell=False, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, )

    return {line.split('=')[0]: int(line.split('=')[1]) for line in result.stdout.split()}


def ffmpeg_get_iframe_times(source, duration):

    result = subprocess.run(["ffmpeg", "-i", source, "-an", "-vf", "setpts=PTS-STARTPTS,select='eq(pict_type,I)',showinfo", "-f", "NULL", "-"],
                            shell=False, encoding='utf-8', capture_output=True)

    match_lines = [line for line in result.stderr.split(
        '\n') if "showinfo" in line and "iskey:1" in line]

    iframes = [0.0]
    durations = []
    lastiframe = 0.0

    for match_line in match_lines:
        match_line_pts = re.search(r'(?:pts_time:)([0-9.]*)', match_line)

        iframe = float(match_line_pts.group(1))
        iframes.append(iframe)
        durations.append(iframe - lastiframe)
        lastiframe = iframe

    durations.append(duration - lastiframe)


def ffmpeg_get_loudness(source):

    result = subprocess.run(["ffmpeg", "-i", source, "-vn", "-af", "loudnorm=print_format=json", "-f", "NULL", "-"],
                            shell=False, encoding='utf-8', capture_output=True)
    result.check_returncode()

    results_text = re.search(r'\{(.*?)\}', result.stderr, re.S).group()
    results_json = json.loads(results_text)

    return float(results_json['input_i']), float(results_json['input_tp'])


def ffmepg_scene_detect(source, threshold, duration):
    result = subprocess.run(["ffmpeg", "-i", local_source, "-an", "-vf", "setpts=PTS-STARTPTS,select='gt(scene,0.5)',showinfo", "-f", "NULL", "-"],
                            shell=False, encoding='utf-8', capture_output=True)

    return parse_scene_detect(result.stderr, duration)


def ffmpeg_run_vmaf(source, transcode, ss, t, dest_fps, dest_size):

    args = ["ffmpeg", "-ss", str(ss), "-i", source, "-t", str(t),
            "-ss", str(ss), "-i", transcode, "-t", str(t), "-an",
            "-filter_complex", "[0:v]fps=" + str(dest_fps) + ",scale=" + str(dest_size['width']) + ":" + str(dest_size['height']) + ",setpts=PTS-STARTPTS[source];[1:v]setpts=PTS-STARTPTS[trancoded];[source][trancoded]libvmaf", "-f", "NULL", " -"]

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    separator = " "
    return parse_vmaf(result.stderr), separator.join(args)


def ffmpeg_run_vmaf_full(source, transcode, log_path, dest_fps, dest_size, subsample=1):

    if not log_path.endswith(".json"):
        raise Exception('log_path must end in .json: {}'.format(log_path))

    args = ["ffmpeg", "-i", source, "-i", transcode, "-an",
            "-filter_complex", "[0:v]fps=" + str(dest_fps) + ",scale=" + str(dest_size['width']) + ":" + str(dest_size['height']) + ",setpts=PTS-STARTPTS[source];[1:v]setpts=PTS-STARTPTS[trancoded];[source][trancoded]libvmaf=log_fmt=json:log_path=" + log_path + ":n_subsample=" + str(subsample), "-f", "NULL", " -"]

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    result.check_returncode()

    return parse_vmaf_json(log_path)


def parse_scene_detect(ffmepgstderr, duration):
    match_lines = [line for line in ffmepgstderr.split(
        '\n') if "showinfo" in line and "pts_time" in line]

    scenes = [0.0]
    durations = []
    last_scene = 0.0

    for match_line in match_lines:
        match_line_pts = re.search(r'(?:pts_time:)([0-9.]*)', match_line)

        current_scene = float(match_line_pts.group(1))
        scenes.append(current_scene)
        durations.append(current_scene - last_scene)
        last_scene = current_scene

    durations.append(duration - last_scene)

    # if we've got a scene list, add a zero-size scene point at the end
    if(len(scenes) > 1):
        scenes.append(duration)
        durations.append(0)

    return scenes, durations


def parse_black_detect(ffmpegtext):
    matched_lines = [line for line in ffmpegtext.split(
        '\n') if "blackdetect" in line]

    found = re.search(
        r'(?:black_start:)([0-9.]*) (?:black_end:)([0-9.]*)', matched_lines[0])

    black_start = found.group(1)
    black_end = found.group(2)

    return black_start, black_end


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

        return vmaf_score, zip(vmav_frame_times, vmaf_frames)


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


def ffmpeg_grab_frame(frame_num, source_path, output_path):
    if not output_path.endswith(".jpg"):
        raise Exception('output_path must end in .jpg: {}'.format(output_path))

    args = ["ffmpeg", "-y", "-i", source_path, "-vf",
            "fps=29.97,scale=1280:720,select = eq(n\\," + str(frame_num) + ")", "-vframes", "1", output_path]

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    result.check_returncode()


def ffmpeg_grab_frames(source_path, frame_nums, output_directory, filename_prefix):
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
            "fps=29.97,select='" + eq_expressions + "',scale=1280:720", "-vsync", "0",  combined_prefix + "%d.jpg"]

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    result.check_returncode()

    return output_files


def ffmpeg_grab_diff_frame(frame_num, source_path, transcode_path, output_path, transcode_fps, dest_size):
    if not output_path.endswith(".jpg"):
        raise Exception('output_path must end in .jpg: {}'.format(output_path))

    args = ["ffmpeg", "-y", "-i", source_path, "-i", transcode_path, "-filter_complex",
            "fps=" + str(transcode_fps) + ",scale=" + str(dest_size['width']) + ":" + str(dest_size['height']) + ",select=eq(n\\," + str(frame_num) + "),blend=all_mode=difference,hue=s=0,eq=gamma=1.7", "-vframes", "1", output_path]

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    if result.returncode:
        print(result.stdout)

    result.check_returncode()


def ffmpeg_grab_diff_frames(frame_nums, source_path, transcode_path, output_directory, transcode_fps, dest_size):
    if not os.path.isdir(output_directory):
        raise Exception(
            'output_directory must be a directory: {}'.format(output_directory))
    filename_prefix = "diff_"
    filepath_prefix = os.path.join(output_directory, filename_prefix)

    eq_expressions = ""
    output_files = []
    for idx, frame_num in enumerate(frame_nums):
        output_files.append(filename_prefix + str(idx + 1) + ".png")
        eq_expressions += "eq(n\\," + str(frame_num) + ")"
        if idx < len(frame_nums) - 1:
            eq_expressions += "+"

    args = ["ffmpeg", "-y", "-i", source_path, "-i", transcode_path, "-filter_complex",
            "[0:v]setpts=PTS-STARTPTS,fps=" + str(transcode_fps) + ",select='" + eq_expressions + "',scale=" + str(dest_size['width']) + ":" + str(dest_size['height']) + "[source];\
                [1:v]setpts=PTS-STARTPTS,select='" + eq_expressions + "'[transcode];[source][transcode]blend=all_mode=difference,hue=s=0,eq=gamma=1.7", "-vsync", "0", filepath_prefix + "%d.png"]

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    if result.returncode:
        print(result.stdout)

    result.check_returncode()

    return output_files


def ffmpeg_grab_diff_frames_cheap(source_files, destfiles, output_directory):
    if not os.path.isdir(output_directory):
        raise Exception(
            'output_directory must be a directory: {}'.format(output_directory))

    filename_prefix = "diff_"

    output_files = []
    for idx, (source_file, dest_file) in enumerate(zip(source_files, destfiles)):

        # relative version for reports
        file_name = filename_prefix + str(idx + 1) + ".png"
        output_files.append(file_name)

        filepath = os.path.join(output_directory, file_name)

        args = ["ffmpeg", "-y", "-i", source_file, "-i", dest_file, "-vf",
                "setpts=PTS-STARTPTS,blend=all_mode=difference,hue=s=0,eq=gamma=1.7", "-vsync", "0", filepath]

        result = subprocess.run(
            args,  shell=False, encoding='utf-8', capture_output=True)

        if result.returncode:
            print(result.stdout)

        result.check_returncode()

    return output_files


def ffmpeg_htile_images(source_frame_path,
                        transcode_frame_path, diff_frame_path, output_path):
    args = ["ffmpeg", "-y", "-i", source_frame_path, "-i", transcode_frame_path, "-i", diff_frame_path, "-filter_complex",
            "hstack=inputs=3",  output_path]

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    if result.returncode:
        print(result.stderr)

    result.check_returncode()


def ffmpeg_grab_frame_triptychs_slow(frame_nums, source_path, transcode_path, results_directory, transcode_fps, dest_size):

    timecode = Timecode(transcode_fps, '00:00:00:00')
    return_frames = []
    source_frames = ffmpeg_grab_frames(
        local_source, frame_nums, results_directory, "source_")
    transcode_frames = ffmpeg_grab_frames(
        local_transcode, frame_nums, results_directory, "transcode_")

    diff_frames = ffmpeg_grab_diff_frames(
        frame_nums, source_path, transcode_path, results_directory, transcode_fps, dest_size)

    for (frame_num, source, transcode, diff) in zip(frame_nums, source_frames, transcode_frames, diff_frames):

        frame_time = timecode.tc_to_string(*timecode.frames_to_tc(frame_num))
        return_path = frame_time.replace(":", "-") + ".jpg"
        return_abs_path = os.path.join(
            results_directory, return_path)

        source_full_path = os.path.join(
            results_directory, source)
        transcode_full_path = os.path.join(
            results_directory, transcode)
        diff_full_path = os.path.join(
            results_directory, diff)

        ffmpeg_htile_images(source_full_path,
                            transcode_full_path, diff_full_path, return_abs_path)

        os.remove(source_full_path)
        os.remove(transcode_full_path)
        os.remove(diff_full_path)

        return_frames.append(return_path)

    return return_frames


def ffmpeg_grab_frame_triptychs_fast(frame_nums, source_path, transcode_path, results_directory, transcode_fps, dest_size):
    if not os.path.isdir(results_directory):
        raise Exception(
            'output_directory must be a directory: {}'.format(results_directory))

    filename_prefix = "frame_"
    filepath_prefix = os.path.join(results_directory, filename_prefix)

    eq_expressions = ""
    output_files = []
    for idx, frame_num in enumerate(frame_nums):
        output_files.append(filename_prefix + str(idx + 1) + ".jpg")
        eq_expressions += "eq(n\\," + str(frame_num) + ")"
        if idx < len(frame_nums) - 1:
            eq_expressions += "+"

    args = ["ffmpeg", "-y", "-i", source_path, "-vsync", "drop", "-i", transcode_path, "-filter_complex",
            "[0:v]setpts=PTS-STARTPTS,fps=" + str(transcode_fps) + ",select='" + eq_expressions + "',scale=" + str(dest_size['width']) + ":" + str(dest_size['height']) + ",split[source0][source1];\
                    [1:v]setpts=PTS-STARTPTS,select='" + eq_expressions + "',split[transcode0][transcode1];[source0][transcode0]blend=all_mode=difference,hue=s=0,eq=gamma=1.7[diff];[source1][transcode1][diff]hstack=inputs=3", "-vsync", "0", filepath_prefix + "%d.jpg"]

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    if result.returncode:
        print(result.stderr)
        # clean up stuff we created
        for file in output_files:
            os.remove(file)

    result.check_returncode()

    return output_files


def write_html_report(report_data, report_path):
    jinja_loader = FileSystemLoader('templates')
    jinja_env = Environment(loader=jinja_loader)
    jinja_temp = jinja_env.get_template("template.html")
    output = jinja_temp.render(data=report_data)
    print(output)
    with open(report_path, "w") as f:
        f.write(output)


def find_valleys(frame_scores, cutoff_index, threshold=20.0, dist=500):
    inverted_scores = [100.0 + (regular_score * -1)
                       for regular_score in frame_scores]

    found_peaks = find_peaks(inverted_scores, threshold, distance=dist)
    indices = [idx for idx in found_peaks[0] if idx < cutoff_index]
    print(indices)
    return indices


def test_transcode_loudness(loudness, true_peak):
    test_result = {}
    test_result['category'] = "Loudness"
    test_result['rule'] = "Loudness should be <= -24 LKFS +/-2.  True-Peak should be <= -2 dBTP"
    test_result['inputs'] = "Integrated (program) loudness: " + str(
        loudness) + " LKFS.  True-Peak: " + str(true_peak) + " dBTP"
    if loudness >= -26 and loudness <= -22 and true_peak <= -2:
        test_result['result'] = "PASSED"
    else:
        test_result['result'] = "FAILED"

    return test_result


def test_source_loudness(loudness, true_peak):
    test_result = {}
    test_result['category'] = "Loudness"
    test_result['rule'] = "Source loudness should be greater than -Inf"
    test_result['inputs'] = "Integrated (program) loudness: " + str(
        loudness) + " LKFS.  True-Peak: " + str(true_peak) + " dBTP"
    if loudness > float('-inf'):
        test_result['result'] = "PASSED"
    else:
        test_result['result'] = "FAILED"

    return test_result


def test_audio_video_duration_match(audio_duration, video_duration):
    test_result = {}
    test_result['category'] = "A/V Sync"
    test_result['rule'] = "Audio duration and video duration should be equal (+/- 2 seconds)"
    test_result['inputs'] = "Audio duration: " + str(
        round(audio_duration, 2)) + "s.  Video duration: " + str(round(video_duration, 2)) + "s.  Diff: " + str(round(abs(audio_duration - video_duration), 2)) + "s."
    if abs(audio_duration - video_duration) < 2.0:
        test_result['result'] = "PASSED"
    else:
        test_result['result'] = "FAILED"

    return test_result


def test_video_expected_duration_match(expected_duration, actual_duration):
    test_result = {}
    test_result['category'] = "A/V Sync"
    test_result[
        'rule'] = "Transcoded VIDEO duration and expected duration should match (+/- 2 seconds)"
    test_result['inputs'] = "Video duration: " + str(round(
        actual_duration, 2)) + "s.  Expected: " + str(round(expected_duration, 2)) + "s. Diff: " + str(round(abs(expected_duration - actual_duration), 2)) + "s."
    if abs(expected_duration - actual_duration) < 2.0:
        test_result['result'] = "PASSED"
    else:
        test_result['result'] = "FAILED"

    return test_result


def test_audio_expected_duration_match(expected_duration, actual_duration):
    test_result = {}
    test_result['category'] = "A/V Sync"
    test_result[
        'rule'] = "Transcoded AUDIO duration and expected duration should match  (+/- 2 seconds)"
    test_result['inputs'] = "Audio duration: " + str(round(
        actual_duration, 2)) + "s.  Expected: " + str(round(expected_duration, 2)) + "s. Diff: " + str(round(abs(expected_duration - actual_duration), 2)) + "s."
    if abs(expected_duration - actual_duration) < 2.0:
        test_result['result'] = "PASSED"
    else:
        test_result['result'] = "FAILED"

    return test_result


def test_program_VMAF(vmaf_score):
    test_result = {}
    test_result['category'] = "Video Quality"
    test_result['rule'] = "VMAF score compared to source should be >= 75"
    test_result['inputs'] = "Program VMAF Score: " + str(vmaf_score)
    if vmaf_score >= 75.0:
        test_result['result'] = "PASSED"
    else:
        test_result['result'] = "FAILED"

    return test_result


def write_dict_to_json(dict, output_path):
    pretty_string = json.dumps(dict, indent=4)
    with open(output_path, 'w') as fp:
        fp.write(pretty_string)
        fp.close()


def create_quality_report(media_id, local_source, local_transcode, results_directory, subsample):

    report_data = {}
    test_data = []
    report_data['title'] = media_id
    report_data['source_file'] = local_source
    report_data['transcode_file'] = local_transcode

    source_dimensions = ffprobe_get_frame_size(local_transcode)
    transcode_dimensions = ffprobe_get_frame_size(local_transcode)
    transcode_framerate = ffprobe_get_framerate(local_transcode)

    source_video_duration = ffprobe_get_video_duration(local_source)
    source_audio_duration = ffprobe_get_audio_duration(local_source)
    transcode_video_duration = ffprobe_get_video_duration(local_transcode)
    transcode_audio_duration = ffprobe_get_audio_duration(local_transcode)
    test_data.append(test_video_expected_duration_match(
        source_video_duration, transcode_video_duration))
    test_data.append(test_audio_expected_duration_match(
        source_audio_duration, transcode_audio_duration))
    test_data.append(test_audio_video_duration_match(
        transcode_audio_duration, transcode_video_duration))

    timecode = Timecode(transcode_framerate, '00:00:00;00')

    source_loudness, source_true_peak = ffmpeg_get_loudness(local_source)
    test_data.append(test_source_loudness(source_loudness, source_true_peak))

    transcode_loudness, transcode_true_peak = ffmpeg_get_loudness(
        local_transcode)
    test_data.append(test_transcode_loudness(
        transcode_loudness, transcode_true_peak))

    report_data['test_data'] = test_data

    json_path = media_id + "_vmaf.json"

    scenes = []
    # scenes, durations = ffmepg_scene_detect(
    #    local_source, 0.5, source_duration)

    # save time if we have this file from a previous run -- for debugging
    if os.path.exists(os.path.join(results_directory, json_path)):
        score, frames = parse_vmaf_json(
            os.path.join(results_directory, json_path))
    else:
        score, frames = ffmpeg_run_vmaf_full(local_source, local_transcode, os.path.join(
            results_directory, json_path), transcode_framerate, transcode_dimensions, subsample)

    test_data.append(test_program_VMAF(score))

    frame_nums, frame_scores = zip(*frames)

    lowest_frames = find_valleys(
        frame_scores, (source_video_duration * float(timecode.framerate))/subsample)

    image_path = "vmaf.png"
    plot_vmaf_graph(os.path.join(results_directory, image_path),
                    frame_nums, frame_scores, source_video_duration, lowest_frames, timecode)

    low_frames = []

    try:
        frame_triptychs = ffmpeg_grab_frame_triptychs_fast(
            lowest_frames, local_source, local_transcode, results_directory, transcode_framerate, transcode_dimensions)
    except:
        frame_triptychs = ffmpeg_grab_frame_triptychs_slow(
            lowest_frames, local_source, local_transcode, results_directory, transcode_framerate, transcode_dimensions)

    for idx, low_frame in enumerate(lowest_frames):
        low_frame_dict = {}
        low_frame_dict['index'] = 1 + idx
        low_frame_dict['frame_grab'] = frame_triptychs[idx]
        low_frame_dict['frame_time'] = timecode.tc_to_string(
            *timecode.frames_to_tc(low_frame))
        low_frame_dict['frame_vmaf'] = frame_scores[low_frame]
        low_frames.append(low_frame_dict)

    report_data['vmaf_score'] = score
    report_data['vmaf_graph'] = image_path
    duration_tc = timecode.frames_to_tc(
        timecode.float_to_tc(source_video_duration))
    report_data['duration'] = timecode.tc_to_string(*duration_tc)
    report_data['full_vmaf_json'] = json_path
    report_data['low_frames'] = low_frames

    # write the unformatted json version of our results
    write_dict_to_json(report_data, os.path.join(
        results_directory, media_id + "_QC.json"))

    # write the formatted html version of our results

    write_html_report(report_data, os.path.join(
        results_directory, media_id + "_quality.html"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        local_source = sys.argv[1]
        local_transcode = sys.argv[2]
        media_id = sys.argv[3]
        subsample = sys.argv[4]
    else:
        # hardcoded for debugging
        # in the future return an error instead
        media_id = "b9adbba5-9607-48a6-8f12-19338fbba09a_Subsample"
        # media_id = "3c3f88a8-2c51-4996-b449-f8a8052b054d"
        local_source = "/Users/kcorcoran/Downloads/StrangeInheritance_QualityTesting/STRANGE_INHERITANCE_S04_E26_AVOD.mov"
        local_transcode = "/Users/kcorcoran/Downloads/StrangeInheritance_QualityTesting/Strange_Inheritence_720p_Fixed.mp4"
        # local_transcode = "/Users/kcorcoran/Downloads/StrangeInheritance_QualityTesting/Strange_Inheritence_720p_BadSync.mp4"
        # media_id = "Netflix_WindAndNature_4096x2160_60fps_ProResHQ_Quicktime"
        # local_source = "/Users/kcorcoran/Downloads/Netflix_WindAndNature_4096x2160_60fps_8bit_420.y4m"
        # local_transcode = "/Users/kcorcoran/Downloads/Netflix_WindAndNature_4096x2160_60fps_ProResHQ.mov"
        subsample = 4

    # create the results directory
    results_directory = expanduser('~') + "/video_quality/"
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)
    work_directory = expanduser('~') + "/video_quality/" + media_id
    if not os.path.isdir(work_directory):
        os.mkdir(work_directory)

    # sys.argv[1], sys.argv[2]
    create_quality_report(media_id, local_source,
                          local_transcode, work_directory, subsample)
