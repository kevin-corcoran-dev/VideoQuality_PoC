import matplotlib.pyplot as plt
from fractions import Fraction
import numpy as np
from scipy.signal import find_peaks
from timecode import Timecode
import matplotlib.lines as mlines


def plot_vmaf_graph(output_path, frame_nums, frame_scores, source_duration, lowest_values, fps, title=""):

    timecode = Timecode(fps, '00:00:00:00')

    plt.title(title)

    # generate major tics based on evenly divided time

    chosen_frames = np.linspace(0, len(frame_nums) - 1, 20)
    ticframes = [frame_nums[int(i)] for i in chosen_frames]
    ticlabels = [timecode.tc_to_string(
        *timecode.frames_to_tc(ticframe)) for ticframe in ticframes]
    plt.xticks(ticframes, ticlabels, rotation='vertical')

    plt.plot(frame_nums, frame_scores)

    ax = plt.axes()

    if lowest_values != None:
        style = dict(size=10, color='gray')
        # label the valleys
        for idx, lowval in enumerate(lowest_values):
            ax.text(lowval, frame_scores[lowval] - 5, str(idx + 1), **style)

    ax.set_xticks(ticframes, minor=True)

    ax.grid()

    plt.ylabel('vmaf score')
    plt.ylim(0, 100)
    plt.xlabel('time')
    plt.subplots_adjust(bottom=0.3)
    plt.gcf().set_size_inches(9, 5)
    plt.savefig(output_path)
    plt.clf()


def plot_multi_vmaf_timegraph(output_path, frame_nums, baseline_frame_scores, variant_list, target_bitrate, source_duration, fps):

    timecode = Timecode(fps, '00:00:00:00')

    title = " Effect of Bitrate Changes on VMAF over Asset Duration"
    plt.suptitle(title, fontsize=14, color='blue')
    plt.title("Blue line is VMAF/time of existing VOD transcode (4sec GOPs).  Gray lines are quality/time for 2sec GOPs at various bitrates",
              fontsize=7, color='black')

    upper = max(baseline_frame_scores)
    lower = min(baseline_frame_scores)

    higher_bitrate_scores = [variant['vmaf_frame_scores']
                             for variant in variant_list if variant['bitrate'] > target_bitrate]

    lower_bitrate_scores = [variant['vmaf_frame_scores']
                            for variant in variant_list if variant['bitrate'] < target_bitrate]

    # generate lighter lineshades
    higher_lineshades = np.linspace(0.7, 0.9, len(higher_bitrate_scores))
    for idx, frame_scores in enumerate(higher_bitrate_scores):
        upper = max(upper, max(frame_scores))
        lower = min(lower, min(frame_scores))
        plt.plot(frame_nums, frame_scores, str(higher_lineshades[idx]))

    lower_lineshades = np.linspace(0.9, 0.7, len(lower_bitrate_scores))
    for idx, frame_scores in enumerate(lower_bitrate_scores):
        upper = max(upper, max(frame_scores))
        lower = min(lower, min(frame_scores))
        # plt.plot(frame_nums, frame_scores, str(1 - (0.1 * (1 + idx))))
        plt.plot(frame_nums, frame_scores, str(lower_lineshades[idx]))

    plt.plot(frame_nums, baseline_frame_scores)

    # generate major tics based on evenly divided time

    chosen_frames = np.linspace(0, len(frame_nums) - 1, 20)
    ticframes = [frame_nums[int(i)] for i in chosen_frames]
    ticlabels = [timecode.tc_to_string(
        *timecode.frames_to_tc(ticframe)) for ticframe in ticframes]

    plt.xticks(ticframes, ticlabels, rotation='vertical')

    ax = plt.axes()
    # style = dict(size=10, color='gray')
    # # label the valleys
    # for idx, lowval in enumerate(lowest_values):
    #     ax.text(lowval, frame_scores[lowval] - 5, str(idx + 1), **style)

    ax.set_xticks(ticframes, minor=True)

    ax.grid()

    plt.ylabel('vmaf score')
    plt.ylim(lower, upper)
    plt.xlabel('time')
    plt.subplots_adjust(bottom=0.3)
    plt.gcf().set_size_inches(15, 5)
    plt.savefig(output_path)
    plt.clf()


def plot_three_vmaf_timegraph(output_path, frame_nums, baseline_frame_scores, variant_list, target_bitrate, recommended_bitrate, source_duration, fps):

    timecode = Timecode(fps, '00:00:00:00')

    upper = max(baseline_frame_scores)
    lower = min(baseline_frame_scores)

    baseline_label = "Existing " + \
        str(target_bitrate) + "bps VOD Transcode (4sec GOP)"
    identical_label = "Identical " + \
        str(target_bitrate) + "bps but with 2sec GOP"
    recommended_label = str(
        target_bitrate) + "bps transcode 2 sec GOP that should match VOD quality"

    # sift through all of the test variants and find the variant that matches the reference bitrate:
    same_bitrate_variant = [variant['vmaf_frame_scores']
                            for variant in variant_list if variant['bitrate'] == target_bitrate][0]
    recommended_bitrate_variant = [variant['vmaf_frame_scores']
                                   for variant in variant_list if variant['bitrate'] == recommended_bitrate][0]

    title = " VOD Transcode (4sec GOP) VMAF vs. Matching Bitrate (2sec GOP) Over Asset Duration"
    plt.title(title, fontsize=14, color='blue')

    upper = max(upper, max(same_bitrate_variant))
    lower = min(lower, min(same_bitrate_variant))
    plt.plot(frame_nums, same_bitrate_variant, color='r',
             label=identical_label)

    upper = max(upper, max(recommended_bitrate_variant))
    lower = min(lower, min(recommended_bitrate_variant))
    plt.plot(frame_nums, recommended_bitrate_variant, color='g',
             label=recommended_label)

    upper = max(baseline_frame_scores)
    lower = min(baseline_frame_scores)
    plt.plot(frame_nums, baseline_frame_scores, color='k',
             label=baseline_label)

    # generate major tics based on evenly divided time

    chosen_frames = np.linspace(0, len(frame_nums) - 1, 20)
    ticframes = [frame_nums[int(i)] for i in chosen_frames]
    ticlabels = [timecode.tc_to_string(
        *timecode.frames_to_tc(ticframe)) for ticframe in ticframes]

    plt.xticks(ticframes, ticlabels, rotation='vertical')

    ax = plt.axes()
    # style = dict(size=10, color='gray')
    # # label the valleys
    # for idx, lowval in enumerate(lowest_values):
    #     ax.text(lowval, frame_scores[lowval] - 5, str(idx + 1), **style)

    ax.set_xticks(ticframes, minor=True)

    ax.grid()

    red_line = mlines.Line2D([], [], color='k', label=baseline_label)
    yellow_line = mlines.Line2D([], [], color='r', label=identical_label)
    green_line = mlines.Line2D([], [], color='g', label=recommended_label)
    plt.legend(handles=[red_line, yellow_line,
                        green_line], loc='lower right')

    plt.ylabel('vmaf score')
    plt.ylim(lower, upper)
    plt.xlabel('time')
    plt.subplots_adjust(bottom=0.3)
    plt.gcf().set_size_inches(15, 5)
    plt.savefig(output_path)
    plt.clf()


def plot_vmaf_vs_bitrate(output_path, template_name, metric_name, reference_bitrate, reference_vmaf, bitrates, variants):

    bitrate_vmafs = [variant[metric_name] for variant in variants]

    if metric_name == 'ssim':
        lower = min(bitrate_vmafs) - 0.01
        upper = min(1.0, 0.01 + max(bitrate_vmafs))
    else:
        lower = round(min(bitrate_vmafs) - 1)
        upper = round(1 + max(bitrate_vmafs))

    title = template_name + " Bitrate vs. " + metric_name + " (2-second GOPs)"
    plt.suptitle(title, fontsize=14, color='blue')
    plt.title("Red cross is quality/bitrate for existing VOD transcode (4sec GOPs).  Blue line is quality/bitrate for 2sec GOPs", fontsize=7, color='black')
    plt.plot(bitrates, bitrate_vmafs)
    plt.axvline(reference_bitrate, color='r')
    plt.axhline(reference_vmaf, color='r')

    ticlabels = [str(bitrate) for bitrate in bitrates]
    plt.xticks(bitrates, ticlabels, rotation='vertical')

    ax = plt.axes()

    ax.grid()

    plt.ylabel(metric_name + " score")
    plt.ylim(lower, upper)
    plt.xlabel('bitrate')
    plt.subplots_adjust(bottom=0.2)
    plt.gcf().set_size_inches(7, 5)
    plt.savefig(output_path)
    plt.clf()
