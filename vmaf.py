import subprocess
import os
import re
import json


def ffmpeg_run_vmaf(source, transcode, ss, t, dest_fps, dest_size):

    args = ["ffmpeg", "-ss", str(ss), "-i", source, "-t", str(t),
            "-ss", str(ss), "-i", transcode, "-t", str(t), "-an",
            "-filter_complex", "[0:v]fps=" + str(dest_fps) + ",scale=" + str(dest_size['width']) + ":" + str(dest_size['height']) + ",setpts=PTS-STARTPTS[source];[1:v]setpts=PTS-STARTPTS[trancoded];[source][trancoded]libvmaf", "-f", "NULL", " -"]

    print(" ".join(args))

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    separator = " "
    return parse_vmaf(result.stderr), separator.join(args)


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
        str_json = f.read()
        str_json = re.sub(r'\bnan\b', 'NaN', str_json)
        vmaf_json = json.loads(str_json)

        vmaf_score = vmaf_json['VMAF score']

        for frame in vmaf_json['frames']:
            vmav_frame_times.append(frame['frameNum'])
            vmaf_frames.append(frame['metrics']['vmaf'])

        return round(vmaf_score, 3), vmav_frame_times, vmaf_frames


def parse_psnr_from_vmaf_json(json_path):

    psnr_frame_scores = []
    psnr_frame_nums = []
    psnr_score = 0

    with open(json_path, 'r') as f:
        str_json = f.read()
        str_json = re.sub(r'\bnan\b', 'NaN', str_json)
        vmaf_json = json.loads(str_json)

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
        str_json = f.read()
        str_json = re.sub(r'\bnan\b', 'NaN', str_json)
        vmaf_json = json.loads(str_json)

        ssim_score = vmaf_json['SSIM score']

        for frame in vmaf_json['frames']:
            ssim_frame_nums.append(frame['frameNum'])
            ssim_frame_scores.append(frame['metrics']['ssim'])

        return ssim_score, ssim_frame_nums, ssim_frame_scores


def run_vmaf_full(source, transcode, log_path, dest_fps, target_width, target_height, subsample=1):

    if not log_path.endswith(".json"):
        raise Exception('log_path must end in .json: {}'.format(log_path))

    args = ["ffmpeg", "-r", str(dest_fps), "-i", source, "-r", str(dest_fps),  "-i", transcode, "-an",
            "-filter_complex", "[0:v]setpts=PTS-STARTPTS,scale=" + str(target_width) + ":" + str(target_height) + ",fifo[source];[1:v]setpts=PTS-STARTPTS,fifo[trancoded];[source][trancoded]libvmaf=log_fmt=json:psnr=1:ssim=1:log_path=" + log_path + ":n_subsample=" + str(subsample), "-f", "NULL", " -"]

    print(" ".join(args))

    result = subprocess.run(
        args,  shell=False, encoding='utf-8', capture_output=True)

    result.check_returncode()

    return parse_vmaf_json(log_path)


def read_or_create_vmaf(source_path, transcode_path, json_path, target_fps, target_width, target_height, subsample):
    if os.path.exists(json_path):
        score, frame_nums, frame_vmafs = parse_vmaf_json(json_path)
    else:
        score, frame_nums, frame_vmafs = run_vmaf_full(
            source_path, transcode_path, json_path, target_fps, target_width, target_height, subsample)

    return score, frame_nums, frame_vmafs
