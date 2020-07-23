import ffmpeg_funcs as ffmpeg_funcs

test_assets = [
    "/Users/kcorcoran/Downloads/StrangeInheritance_QualityTesting/STRANGE_INHERITANCE_S04_E26_AVOD.mov"]


def detect_silent_black(asset):
    result = subprocess.run(["ffmpeg", "-i", source, "-vf", "setpts=PTS-STARTPTS,blackdetect", "-af", "silencedetect" "-f", "NULL", "-"],
                            shell=False, encoding='utf-8', capture_output=True)

    black_match_lines = [line for line in result.stderr.split(
        '\n') if "black_start" in line and "black_end" in line]

    silent_match_lines = [line for line in result.stderr.split(
        '\n') if "silence_end" in line and "silence_duration" in line]

    scenes = [0.0]

    for match_line in match_lines:
        match_line_pts = re.search(r'(?:pts_time:)([0-9.]*)', match_line)

        current_scene = float(match_line_pts.group(1))
        scenes.append(current_scene)

    return scenes


for test_asset in test_assets:
    def scene_detect(source, threshold):
