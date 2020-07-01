# approximation of Titan's ffmpeg command manipulations
# adapted to Python

compare_float_precision = 0.01


def round_to_even(number):
    round_number = round(number)
    if (round_number % 2) == 0:
        return round_number
    else:
        return round_number + 1


def generate_key_frame_placement(fps):
    kfp = ""
    if fps < 26:
        kfp = "-g 96 -keyint_min 48 -flags +cgop"
    else:
        kfp = "-g 120 -keyint_min 60 -flags +cgop"
    return kfp


def generate_custom_key_frame_placement(fps, interval_secs):
    interval_frames = str(int((0.5 + fps) * interval_secs))
    return "-g " + interval_frames + " -keyint_min " + \
        interval_frames + " -flags +cgop"


def generate_fps_filter(fps):
    if fps >= 24.5 and fps < 25.5:
        return "-r 24000/1001"

    elif fps > 59 and fps <= 60:
        return "-r 30000/1001"

    elif fps >= 24 and fps < 24.5:
        return "-r 24"

    elif fps >= 23.5 and fps < 24:
        return "-r 24000/1001"

    elif fps > 29.5 and fps <= 30:
        return "-r 30000/1001"

    elif fps > 30 and fps <= 30.5:
        return "-r 30"

    else:
        return "-r 30000/1001"


def generate_fps_float(fps):
    if fps >= 24.5 and fps < 25.5:
        return 24000/1001

    elif fps > 59 and fps <= 60:
        return 30000/1001

    elif fps >= 24 and fps < 24.5:
        return 24

    elif fps >= 23.5 and fps < 24:
        return 24000/1001

    elif fps > 29.5 and fps <= 30:
        return 30000/1001

    elif fps > 30 and fps <= 30.5:
        return 30

    else:
        return 30000/1001


def generate_audio_drift_video_filter(fps):
    if abs(fps - 25) < compare_float_precision:
        return "setpts=1.04270937604270937604270937604271*PTS"
    else:
        return ""


def generate_video_filter(target_width, target_height, fps, sample_aspect_ratio, deinterlace, original_width, original_height):

    video_filter = ""

    # use original width and height to replace cropped width and height
    crops = [str(original_width), str(original_height), "0", "0"]
    crop_filter = "crop=" + ":".join(crops)

    ratio = sample_aspect_ratio.split(":")
    x = int(ratio[0])
    y = int(ratio[1])

    sar_float = 1
    if x != 0 and y != 0:
        sar_float = x/y

    source_width = original_width * sar_float
    source_height = original_height

    width = 0
    height = 0
    if source_width / source_height >= 16 / 9:
        width = target_width
        height = round_to_even(target_width * source_height / source_width)

    else:
        width = round_to_even(target_height * source_width /
                              source_height)
        height = target_height

    fps_filter = generate_fps_filter(fps)

    # Special Case where when halving the frame rate from 59.94i to 29.97p we will get drift
    # when using just the standard generated fps_filter
    select_filter = generate_select_filter(fps, fps_filter)

    scale_filter = "scale=" + str(width) + ":" + str(height)
    audio_drift_video_filter = generate_audio_drift_video_filter(fps)

    # deinterlce must be before other transformations
    filter_list = [
        deinterlace,
        select_filter,
        audio_drift_video_filter,
        crop_filter,
        scale_filter,
        "setsar=1/1"]

    filter_list = [item for item in filter_list if item != ""]
    video_filter = ",".join(filter_list)

    full_video_filter = fps_filter + " -vf " + video_filter

    return full_video_filter


def generate_select_filter(source_fps, fps_filter):
    if source_fps > 59 and source_fps <= 60 and fps_filter == "-r 30000/1001":
        return "select='mod(n-1\,2)'"
    else:
        return ""
