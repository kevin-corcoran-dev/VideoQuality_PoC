
import numpy as np
import re


def generate_crf_factors(proposed_target):
    new_targets = np.linspace(int(proposed_target/2),
                              int(proposed_target * 2), 10)
    return [int(target) for target in new_targets]


def generate_crf_range(low, high, steps):
    new_targets = np.linspace(high, low, steps)
    return [int(target) for target in new_targets]


def generate_alternate_bitrates(original_target):
    new_targets = np.linspace(int(original_target/2),
                              int(original_target * 2), 10)
    return [int(target) for target in new_targets]


def change_crf(template, new_crf):
    new_template = template

    original_crf = re.search(
        r'(?:-crf )([0-9]*)', new_template, re.S)[0]
    new_template = new_template.replace(original_crf, "-crf " + str(new_crf))

    return new_template


def get_crf(template):

    crf = re.search(
        r'(?:-crf )([0-9]*)', template, re.S)[1]
    return int(crf)


def change_bitrate(template, new_bitrate, new_maxrate=0, new_minrate=0, new_bufsize=0):

    new_template = template

    if new_maxrate == 0:
        new_maxrate = new_bitrate

    if new_minrate == 0:
        new_minrate = new_bitrate

    if new_bufsize == 0:
        new_bufsize = new_bitrate

    # search and replace the bitrate
    original_bitrate = re.search(
        r'(?:-b:v )([0-9.kM]*)', new_template, re.S)[0]
    new_template = new_template.replace(
        original_bitrate, "-b:v " + str(new_bitrate))

    # search and replace maxrate
    original_maxrate = re.search(
        r'(?:-maxrate )([0-9.kM]*)', new_template, re.S)[0]
    new_template = new_template.replace(
        original_maxrate, "-maxrate " + str(new_maxrate))

    # search and replace minrate
    original_minrate = re.search(
        r'(?:-minrate )([0-9.kM]*)', new_template, re.S)[0]
    new_template = new_template.replace(
        original_minrate, "-minrate " + str(new_maxrate))

    # search and replace the buffer
    original_buffersize = re.search(
        r'(?:bufsize )([0-9.]*k)', new_template, re.S)[0]

    new_template = new_template.replace(
        original_buffersize, "bufsize " + str(new_bufsize))

    return new_template
