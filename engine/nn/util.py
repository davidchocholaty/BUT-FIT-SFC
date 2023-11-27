# Project: Demonstration of backpropagation learning - basic algorithm and selected optimizer
# Author: David Chocholaty <xchoch09@stud.fit.vutbr.cz>
# File: util.py

import numpy as np

# The source code of the following file is taken over from the implementation of the Tensorgrad library.
# Source: https://github.com/hkxIron/tensorgrad/blob/6098d54eeeeeebf69ee89a2dcb0a7d8b60b95c16/tensorgrad/util.py


def get_repeat_axis(left_shape, right_shape):
    len_left = len(left_shape)
    len_right = len(right_shape)
    left_not_repeat = len_left - len_right
    return left_not_repeat, tuple(np.arange(abs(left_not_repeat)))


def accumulative_add_by_shape(is_accumulate_not_repeat, repeat_axis, accumulate_add, to_add):
    if is_accumulate_not_repeat >= 0:
        accumulate_add += to_add
    else:
        accumulate_add += to_add.sum(axis=repeat_axis, keepdims=False)
