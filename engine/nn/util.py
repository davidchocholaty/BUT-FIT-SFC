import numpy as np


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
