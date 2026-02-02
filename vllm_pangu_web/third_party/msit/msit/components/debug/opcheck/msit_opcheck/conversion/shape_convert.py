# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Union, Tuple, List
from functools import reduce

import numpy as np

from components.debug.common import logger
from msit_opcheck.utils import align, ceil_div, lcm

BLOCK_SIZE = 16
PAD_C0_MAPPING = {
    "int64": 4,
    "uint64": 4,
    "float32": 16,
    "int32": 16,
    "uint32": 16,
    "float16": 16,
    "int16": 16,
    "uint16": 16,
    "int8": 32,
    "uint8": 32,
    "bool": 32,
    "uint1": 256,
}


def align_factor(dtype: str = "float16"):
    return PAD_C0_MAPPING.get(dtype, 16)


def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]


def determine_c0(dtype, target_shape: Union[List, Tuple] = None) -> int:
    if not isinstance(dtype, str) and hasattr(dtype, 'name'):
        dtype = getattr(dtype, 'name')
    if target_shape and target_shape[-1] > 0:
        return target_shape[-1]
    else:
        return align_factor(dtype)


def _calculate_group(cin, cout, groups, c0):
    cin_ori, cout_ori = cin // groups, cout // groups
    mag_factor0 = lcm(cin_ori, c0) // cin_ori
    mag_factor1 = lcm(cout_ori, BLOCK_SIZE) // cout_ori
    mag_factor = min(lcm(mag_factor0, mag_factor1), groups)

    cin_g = align(mag_factor * cin_ori, c0)
    cout_g = align(mag_factor * cout_ori, BLOCK_SIZE)

    group_dict = {
        "real_g": ceil_div(groups, mag_factor),
        "mag_factor": mag_factor,
        "cin_g": cin_g,
        "cin1_g": cin_g // c0,
        "cout_g": cout_g,
        "cout1_g": cout_g // BLOCK_SIZE,
        "groups": groups,
        "cin_ori": cin_ori,
        "cout_ori": cout_ori
    }
    logger.debug('cin:%d, cout:%d, groups:%d, group_dict:' % (cin, cout, groups), group_dict)
    return group_dict


def is_nchw_like(shape, format_: str) -> bool:
    return len(shape) == 4 and len(format_) == 4 and all([c in format_ for c in "NCHW"])


def is_ndchw_like(shape, format_: str) -> bool:
    return len(shape) == 5 and len(format_) == 5 and all([c in format_ for c in "NDCHW"])


def nd_shape2fhd_shape(nd_shape, nd_format: str = "NCHW", dtype: str = "float16",
                       fhd_shape: Union[List, Tuple] = None) -> Tuple:
    if not is_nchw_like(nd_shape, nd_format):
        raise RuntimeError(f"shape: {nd_shape} of format {nd_format} is not NCHW-like.")
    c0 = determine_c0(dtype, fhd_shape)
    n, c = nd_shape[nd_format.index("N")], nd_shape[nd_format.index("C")]
    h, w = nd_shape[nd_format.index("H")], nd_shape[nd_format.index("W")]
    c1 = ceil_div(c, c0)
    return n, c1, h, w, c0


def nd_shape2nz_shape(nd_shape: Union[List, Tuple], dtype: str = "float16",
                      nz_shape: Union[List, Tuple] = None) -> Tuple:
    m, n = nd_shape[-2:]
    m0 = 16
    n0 = determine_c0(dtype, nz_shape)
    m1 = ceil_div(m, m0)
    n1 = ceil_div(n, n0)
    return tuple(nd_shape[:-2] + (n1, m1, m0, n0))


def fhd2nd(data, nd_shape, nd_format: str = "NCHW"):
    if not is_nchw_like(nd_shape, nd_format):
        raise RuntimeError(f"shape: {nd_shape} of format {nd_format} is not NCHW-like.")
    fhd_shape = data.shape
    pad = 1 + (nd_shape[nd_format.index("C")] - 1) % fhd_shape[-1]
    main_block = data[:, :fhd_shape[1] - 1, :, :, :]  # main block
    tail_block = data[:, fhd_shape[1] - 1, :, :, :pad]  # tail block
    # NC1HWC0 -> NHWC1C0
    main_block = main_block.transpose((0, 2, 3, 1, 4))
    # NHWC1C0 -> NHWC
    main_block = main_block.reshape(main_block.shape[:3] + (-1,))
    # concatenate
    nhwc = np.concatenate((main_block, tail_block), axis=-1)
    if nd_format != "NHWC":
        return nhwc.transpose(("NHWC".index(nd_format[0]), "NHWC".index(nd_format[1]),
                               "NHWC".index(nd_format[2]), "NHWC".index(nd_format[3])))
    return nhwc


def shd2nd(data, nd_shape, nd_format: str = "NDCHW"):
    if not is_ndchw_like(nd_shape, nd_format):
        raise RuntimeError(f"shape: {nd_shape} of format {nd_format} is not NDCHW-like.")
    shd_shape = data.shape
    pad = 1 + (nd_shape[nd_format.index("C")] - 1) % shd_shape[-1]
    main_block = data[:, :, :shd_shape[2] - 1, :, :, :]
    tail_block = data[:, :, shd_shape[2] - 1, :, :, :pad]
    # NDC1HWC0 -> NDHWC1C0
    main_block = main_block.transpose((0, 1, 3, 4, 2, 5))
    # NDHWC1C0 -> NDHWC
    main_block = main_block.reshape(main_block.shape[:4] + (-1,))
    # concatenate
    ndhwc = np.concatenate((main_block, tail_block), axis=-1)
    if nd_format != "NDHWC":
        return ndhwc.transpose(("NDHWC".index(nd_format[0]), "NDHWC".index(nd_format[1]),
                                "NDHWC".index(nd_format[2]), "NDHWC".index(nd_format[3]),
                                "NDHWC".index(nd_format[4])))
    return ndhwc


def nd2fhd(data, nd_format="NCHW", fhd_shape: Union[List, Tuple] = None):
    nd_shape = data.shape
    if not is_nchw_like(nd_shape, nd_format):
        raise RuntimeError(f"shape: {nd_shape} of format {nd_format} is not NCHW-like.")
    if nd_format != "NHWC":
        data = np.transpose(data, axes=(nd_format.index("N"), nd_format.index("H"),
                                           nd_format.index("W"), nd_format.index("C")))  # pivot to NHWC
    nd_shape = data.shape
    c = nd_shape[-1]
    c0 = determine_c0(data.dtype.name, fhd_shape)
    c1c0 = align(c, c0)
    if c1c0 > c:
        zero_block = np.zeros(nd_shape[:3] + (c1c0 - nd_shape[3],), dtype=data.dtype)
        fhd = np.concatenate((data, zero_block), axis=-1).reshape((nd_shape[0], nd_shape[1], nd_shape[2], -1, c0))
    else:
        fhd = data.reshape((nd_shape[0], nd_shape[1], nd_shape[2], -1, c0))
    fhd = np.transpose(fhd, axes=(0, 3, 1, 2, 4))
    return fhd


def nz2nd(data, nd_shape):
    """
    Convert FRACTAL_NZ format to ND format
    (A0, A1, A2, ..., An, N1, M1, M0, N0) -> (A, N1, M1, M0, N0) -> (A, M1, M0, N1, N0)
    (A, M1, M0, N1, N0) -> (A, [0, M1-2], M0, [0, N1-2], N0) == (A, (M1-1), M0, (N1-1), N0)
                                                             -> (A, (M1-1) *M0, (N1-1) *N0)
                        -> (A, [M1-1], [0, pad_m-1], [0, N1-2], N0) == (A, 1, pad_m, (N1-1), N0)
                                                                    -> (A, pad_m, (N1-1) *N0)
                        -> (A, [0, M1-2], M0, [N1-1], [0, pad_n-1]) == (A, (M1-1), M0, 1, pad_n)
                                                                    -> (A, (M1-1) *M0, pad_n)
                        -> (A, [M1-1], [0, pad_m-1], [N1-1], [0, pad_n-1]) == (A, 1, pad_m, 1, pad_n)
                                                                           -> (A, pad_m, pad_n)
    M = (M1-1) *M0 + pad_m
    N = (N1-1) *N0 + pad_n
    (A, (M1-1) *M0, (N1-1) *N0) + (A, pad_m, (N1-1) *N0) -> (A, (M1-1) *M0 + pad_m, (N1-1) *N0) -> (A, M, (N1-1) *N0)
    (A, (M1-1) *M0, pad_n) + (A, pad_m, pad_n) -> (A, (M1-1) *M0 + pad_m, pad_n) -> (A, M, pad_n)
    (A, M, (N1-1) *N0) + (A, M, pad_n) -> (A, M, (N1-1) *N0 + pad_n) -> (A, M, N)
    (A, M, N) -> (A0, A1, A2, ..., An, M, N)
    """
    ori_nd_shape = copy.deepcopy(nd_shape)
    if len(data.shape) == 4:
        data = np.reshape(data, (1,) + data.shape)
    nd_shape = (1,) + tuple(nd_shape)
    data_shape = data.shape
    m, n = nd_shape[-2:]
    n1, m1 = data_shape[-4:-2]
    m0, n0 = data_shape[-2:]
    pad_m = 1 + (m - 1) % m0
    pad_n = 1 + (n - 1) % n0
    # (A0, A1, A2, ... , An, N1, M1, M0, N0) -> (A, N1, M1, M0, N0) -> (A, M1, M0, N1, N0)
    data = np.reshape(data, (np.prod(data_shape[:-4]),) + data_shape[-4:]).transpose((0, 2, 3, 1, 4))
    main_block = data[:, :m1 - 1, :, :n1 - 1, :]  # main block
    part_1 = data[:, m1 - 1, :pad_m, :n1 - 1, :]  # part 1
    part_2 = data[:, :m1 - 1, :, n1 - 1, :pad_n]  # part 2
    tail_block = data[:, m1 - 1, :pad_m, n1 - 1, :pad_n]  # tail_block
    # Reshape
    a = data.shape[0]
    main_block = np.reshape(main_block, (a, (m1 - 1) * m0, (n1 - 1) * n0))
    part_1 = np.reshape(part_1, (a, pad_m, (n1 - 1) * n0))
    part_2 = np.reshape(part_2, (a, (m1 - 1) * m0, pad_n))
    tail_block = np.reshape(tail_block, (a, pad_m, pad_n))
    # Concatenate
    main_concat_part1 = np.concatenate((main_block, part_1), axis=1)  # (A, M, (N1-1) *N0)
    part_2_concat_tail = np.concatenate((part_2, tail_block), axis=1)  # (A, M, pad_n)
    nd = np.concatenate((main_concat_part1, part_2_concat_tail), axis=-1)
    # Reshape
    nd = np.reshape(nd, data_shape[:-4]+(m, n))
    nd = np.reshape(nd, ori_nd_shape)

    return nd


def nd2nz(data, nz_shape: Union[List, Tuple] = None):
    """
    Convert ND format to FRACTAL_NZ format
    (A0, A1, A2, ..., An, M, N) -> (A, M, N)
    (A, M, N) -> (A, M, (N1-1) *N0 + pad_n) -> (A, M, (N1-1) *N0) -> (A, (M1-1) *M0, (N1-1) *N0)
                                                                  -> (A, pad_m, (N1-1) *N0)
                                            -> (A, M, pad_n)      -> (A, (M1-1) *M0, pad_n)
                                                                  -> (A, pad_m, pad_n)
    (A, (M1-1) *M0, pad_n)      -> (A, (M1-1), M0, 1, pad_n)   -> (A, [0, M1-2], M0, [N1-1], [0, pad_n-1])
                                                               => (A, [0, M1-2], M0, [N1-1], N0)
    (A, (M1-1) *M0, (N1-1) *N0) -> (A, (M1-1), M0, (N1-1), N0) -> (A, [0, M1-2], M0, [0, N1-2], N0)
                                                              ==> (A, [0, M1-2], M0, N1, N0)
    (A, pad_m, pad_n)           -> (A, 1, pad_m, 1, pad_n)     -> (A, [M1-1], [0, pad_m-1], [N1-1], [0, pad_n-1])
                                                               => (A, [M1-1], [0, pad_m-1], [N1-1], N0)
    (A, pad_m, (N1-1) *N0)      -> (A, 1, pad_m, (N1-1), N0)   -> (A, [M1-1], [0, pad_m-1], [0, N1-2], N0)
                                                              ==> (A, [M1-1], [0, pad_m-1], N1, N0)
    (A, [M1-1], [0, pad_m-1], N1, N0) => (A, [M1-1], M0, N1, N0)
    (A, [0, M1-2], M0, N1, N0)        == (A, [0, M1-2], M0, N1, N0)
                                     ==> (A, M1, M0, N1, N0)
    """
    ori_nd_shape = data.shape
    if len(ori_nd_shape) <= 2:
        data = np.reshape(data, (1,) * (3 - len(ori_nd_shape)) + ori_nd_shape)
    data_shape = data.shape
    nz_shape = nd_shape2nz_shape(data_shape, data.dtype, nz_shape)
    a = np.prod(data_shape[:-2])
    m, n = data_shape[-2:]
    n1, m1 = nz_shape[-4:-2]
    m0, n0 = nz_shape[-2:]
    pad_m = 1 + (m - 1) % m0
    pad_n = 1 + (n - 1) % n0

    data = np.reshape(data, (a, m, n))

    main_concat_part1 = data[:, :, :(n1 - 1) * n0]  # -> (A, M, (N1-1) *N0)
    main_block = main_concat_part1[:, :(m1 - 1) * m0, :]  # (A, (M1-1) *M0, (N1-1) *N0)
    part_1 = main_concat_part1[:, (m1 - 1) * m0:, :]  # (A, pad_m, (N1-1) *N0)

    part_2_concat_tail = data[:, :, (n1 - 1) * n0:]  # -> (A, M, pad_n)
    part_2 = part_2_concat_tail[:, :(m1 - 1) * m0, :]  # (A, (M1-1) *M0, pad_n)
    tail_block = part_2_concat_tail[:, (m1 - 1) * m0:, :]  # (A, pad_m, pad_n)

    main_block = np.reshape(main_block, (a, (m1 - 1), m0, (n1 - 1), n0))  # (A, (M1-1), M0, (N1-1), N0)
    part_1 = np.reshape(part_1, (a, 1, pad_m, (n1 - 1), n0))  # (A, 1, pad_m, (N1-1), N0)
    part_2 = np.reshape(part_2, (a, (m1 - 1), m0, 1, pad_n))  # (A, (M1-1), M0, 1, pad_n)
    tail_block = np.reshape(tail_block, (a, 1, pad_m, 1, pad_n))  # (A, 1, pad_m, 1, pad_n)

    part_2_pad = np.concatenate((part_2,
                                    np.zeros((a, m1-1, m0, 1, n0-pad_n), dtype=data.dtype)), axis=-1)

    main_block_concat_part_2_pad = np.concatenate((main_block, part_2_pad), axis=-2)

    tail_block_pad = np.concatenate((tail_block,
                                        np.zeros((a, 1, pad_m, 1, n0-pad_n), dtype=data.dtype)), axis=-1)

    part_1_concat_tail_block_pad = np.concatenate((part_1, tail_block_pad), axis=-2)

    part_1_concat_tail_block_pad_pad \
        = np.concatenate((part_1_concat_tail_block_pad,
                             np.zeros((a, 1, m0-pad_m, n1, n0), dtype=data.dtype)), axis=2)
    nz = np.concatenate((main_block_concat_part_2_pad, part_1_concat_tail_block_pad_pad), axis=1)
    nz = np.transpose(nz, (0, 3, 1, 2, 4)).reshape(data_shape[:-2]+(n1, m1, m0, n0))
    if len(ori_nd_shape) <= 2:
        nz = np.reshape(nz, nz.shape[-4:])

    return nz


def to_fractal_z(data: np.ndarray, ori_format: str, target_shape: Union[List, Tuple] = None, groups=None):
    data_shape = data.shape
    if not is_nchw_like(data_shape, ori_format):
        raise RuntimeError(f"shape: {data_shape} of format {ori_format} is not NCHW-like.")
    # shape转换： NCHW 或者 NHWC
    n, c = data_shape[ori_format.index("N")], data_shape[ori_format.index("C")]
    h, w = data_shape[ori_format.index("H")], data_shape[ori_format.index("W")]
    if groups is None:
        groups = 1
    c_in = c * groups
    c_out = n
    c0 = determine_c0(data.dtype.name, target_shape)
    group_dict = _calculate_group(c_in, c_out, groups, c0)
    g = group_dict["real_g"]
    ci_ori = group_dict["cin_ori"]
    co_ori = group_dict["cout_ori"]
    cin1_g = group_dict["cin1_g"]
    cou1_g = group_dict["cout1_g"]
    e = group_dict["mag_factor"]
    # Initialization
    out = np.zeros([g * cou1_g * BLOCK_SIZE, cin1_g * c0, h, w], dtype=data.dtype)
    data = data.transpose([ori_format.index("N"), ori_format.index("C"), ori_format.index("H"), ori_format.index("W")])
    for m in range(groups):
        for k in range(co_ori):
            for n in range(0, ci_ori):
                i = m // e
                j = m % e
                out[i * e * co_ori + j * co_ori + k, j * ci_ori + n, :, :] = \
                    data[i * e * co_ori + j * co_ori + k, n, :, :]
    # nchw->FRACTAL_Z
    out = out.reshape((g, cou1_g * BLOCK_SIZE, cin1_g, c0, h, w)).transpose(0, 2, 4, 5, 1, 3)
    out = out.reshape(g * cin1_g * h * w, cou1_g, BLOCK_SIZE, c0)
    return out


def to_fractal_z_c04(data: np.ndarray, ori_format: str, target_shape: Union[List, Tuple] = None, groups=None):
    data_shape = data.shape
    
    n, c = data_shape[ori_format.index("N")], data_shape[ori_format.index("C")]
    h, w = data_shape[ori_format.index("H")], data_shape[ori_format.index("W")]
    if groups is None:
        groups = 1
    c_in = c * groups
    c_out = n
    c0 = determine_c0(data.dtype.name, target_shape)
    group_dict = _calculate_group(c_in, c_out, groups, c0)
    g = group_dict["real_g"]
    ci_ori = group_dict["cin_ori"]
    co_ori = group_dict["cout_ori"]
    cin1_g = group_dict["cin1_g"]
    cou1_g = group_dict["cout1_g"]
    # Initialization
    out = np.zeros([g * cou1_g * BLOCK_SIZE, cin1_g * c0, h, w], dtype=data.dtype)
    data = data.transpose([ori_format.index("N"), ori_format.index("C"), ori_format.index("H"), ori_format.index("W")])
    # NCHW->FractalZ_C04
    for k in range(co_ori):
        for n in range(0, ci_ori):
            out[k, n, :, :] = data[k, n, :, :]
    # NCHW->Fractal_Z
    out = out.reshape((g, cou1_g * BLOCK_SIZE, cin1_g, c0, h, w)).transpose(0, 2, 4, 5, 1, 3)
    out = out.reshape(g * cin1_g * h * w, cou1_g, BLOCK_SIZE, c0)
    out_pad = np.zeros([align(g * cin1_g * h * w, 4), cou1_g, BLOCK_SIZE, c0], dtype=data.dtype)
    for i in range(g * cin1_g * h * w):
        out_pad[i, :, :, :] = out[i, :, :, :]
    cin_outer = ceil_div(g * ceil_div(c_in, 4) * 4 * h * w, BLOCK_SIZE)
    fractal_z_c04_res = np.zeros([cin_outer, cou1_g, BLOCK_SIZE, c0], dtype=data.dtype)
    for k in range(cin_outer):
        for n in range(cou1_g):
            for cou0 in range(BLOCK_SIZE):
                for cin0 in range(c0):
                    fractal_z_c04_res[k, n, cou0, cin0] = out_pad[k * 4 + cin0 // 4, n, cou0, cin0 % 4]
    return fractal_z_c04_res


def to_fractal_z_3d(data: np.ndarray, ori_format: str, target_shape: Union[list, tuple] = None, groups=None):
    data_shape = data.shape
    # shape转换：NCDHW 或者 NDHWC
    n, c = data_shape[ori_format.index("N")], data_shape[ori_format.index("C")]
    h, w = data_shape[ori_format.index("H")], data_shape[ori_format.index("W")]
    d = data_shape[ori_format.index("D")]
    if groups is None:
        groups = 1
    fmap_c = c * groups
    out_c = n
    c0 = determine_c0(data.dtype.name, target_shape)
    group_dict = _calculate_group(fmap_c, out_c, groups, c0)
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout_g = group_dict["cout_g"]
    mag_factor = group_dict["mag_factor"]
    if mag_factor == 0:
        raise ZeroDivisionError("Parameter mag_factor is Zero, Please check!")
    cout1_g = group_dict["cout1_g"]
    weight_group = np.zeros((real_g, d, cin1_g, h, w, cout_g, c0), dtype=data.dtype)
    data = data.transpose([ori_format.index("N"), ori_format.index("C"),
                           ori_format.index("D"), ori_format.index("H"), ori_format.index("W")])
    for g in range(groups):
        for ci in range(c):
            for co in range(n // groups):
                try:
                    e = g % mag_factor
                    dst_cin = e * c + ci
                    dst_cout = e * (n // groups) + co
                    src_cout = g * (n // groups) + co
                    weight_group[g // mag_factor, :, dst_cin // c0, :, :, dst_cout, dst_cin % c0] = \
                        data[src_cout, ci, :, :, :]
                except:
                    e = g % mag_factor
                    dst_cin = e * c + ci
                    dst_cout = e * (n // groups) + co
                    src_cout = g * (n // groups) + co
                    logger.error(
                        "================================== Error Detected ======================================="
                    )
                    logger.error("weight_group shape:", weight_group.shape)
                    logger.error("Weight Shape : ", filter_data.shape)
                    logger.error("C0:", co)
                    logger.error("e : ", e)
                    logger.error("dst_cin :", dst_cin)
                    logger.error("dst_cout : ", dst_cout)
                    logger.error("src_cout and Ci", src_cout, "", ci)
                    logger.error("mag_factor : ", mag_factor)
                    raise
    weight_group = weight_group.reshape([real_g * d * cin1_g * h * w, cout1_g, BLOCK_SIZE, c0])
    return weight_group


def to_nc1hwc0(data: np.ndarray, ori_format: str, target_shape: Union[List, Tuple] = None, groups=None):
    ori_shape = data.shape
    if len(ori_shape) > 4:
        raise RuntimeError("Please check original format and original shape: NC1HWC0 transformer doesn't support"
                           f" {len(ori_shape)}D shape")
    c0 = determine_c0(data.dtype.name, target_shape)
    n, c, h, w = 1, 1, 1, 1
    transpose_axis = []
    if "N" in ori_format:
        transpose_axis.append(ori_format.index("N"))
        n = ori_shape[transpose_axis[-1]]
    if "C" in ori_format:
        transpose_axis.append(ori_format.index("C"))
        c = ori_shape[transpose_axis[-1]]
        c1 = ceil_div(c, c0)
    else:
        data = data.reshape(data.shape + (1,))
        transpose_axis.append(len(data.shape) - 1)
        c1 = c0
    if "H" in ori_format:
        transpose_axis.append(ori_format.index("H"))
        h = ori_shape[transpose_axis[-1]]
    if "W" in ori_format:
        transpose_axis.append(ori_format.index("W"))
        w = ori_shape[transpose_axis[-1]]
    data = data.transpose(transpose_axis)
    num_2_padding_in_cin = c1 * c0 - c
    zero_padding_array = np.zeros((n, num_2_padding_in_cin, h, w), dtype=data.dtype)
    data = np.concatenate((data, zero_padding_array), axis=1)
    data = data.reshape((n, c1, c0, h, w)).transpose(0, 1, 3, 4, 2)
    return data


def to_ndc1hwc0(data: np.ndarray, ori_format: str, target_shape: Union[List, Tuple] = None, groups=None):
    ori_shape = data.shape
    n, c = ori_shape[ori_format.index("N")], ori_shape[ori_format.index("C")]
    h, w = ori_shape[ori_format.index("H")], ori_shape[ori_format.index("W")]
    d = ori_shape[ori_format.index("D")]
    c0 = determine_c0(data.dtype.name, target_shape)
    c1 = ceil_div(c, c0)
    data = data.transpose([ori_format.index("N"), ori_format.index("C"),
                           ori_format.index("D"), ori_format.index("H"), ori_format.index("W")])
    num_2_padding_in_cin = c1 * c0 - c
    zero_padding_array = np.zeros((n, num_2_padding_in_cin, d, h, w), dtype=data.dtype)
    data = np.concatenate((data, zero_padding_array), axis=1)
    data = data.reshape((n, c1, c0, d, h, w)).transpose(0, 3, 1, 4, 5, 2)
    return data


def nd_to_fractal_nz(data: np.ndarray, ori_format: str, target_shape: Union[List, Tuple] = None, groups=None):
    ori_shape = data.shape
    if len(ori_shape) < 2:
        err_msg = "If you want to convert the ND format to the NZ format, " + \
                  "the shape dimension of the input ND format data must be greater than or equal to 2."
        raise ValueError(err_msg)
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = ((0, 0),) * batch_num
    m0, n0 = 16, determine_c0(data.dtype.name, target_shape)
    m1, n1 = ceil_div(m_ori, m0), ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    data = np.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), 'constant')
    array_trans = gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
    data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    return data


def nd_to_fractal_z(data: np.ndarray, ori_format: str, target_shape: Union[List, Tuple] = None, groups=None):
    ori_shape = data.shape
    if len(ori_shape) != 4:
        err_msg = "If you want to convert the ND format to the fractal_z format, " + \
                  "the shape dimension of the input ND format data must be equal to 4."
        raise ValueError(err_msg)
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = ((0, 0),) * batch_num
    m0, n0 = determine_c0(data.dtype.name, target_shape), 16
    m1, n1 = ceil_div(m_ori, m0), ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    data = np.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), 'constant')
    array_trans = gen_axes_for_transpose(len(data.shape) - 2, [0, 2, 3, 1])
    data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    return data


def update_axis_for_npu_inner_format(ori_shape, axis, input_format, ori_format, reduce_mode=False):
    """
    when format is changed as npu inner format, the axis will be updated
    """
    if input_format in ("NDC1HWC0", "NC1HWC0"):
        ori_shape_len = len(ori_shape) if -2 not in ori_shape else len(ori_format)
        axis = axis % ori_shape_len
        ''' 示例：
        ori axis with N, axis = 0
        ori axis with D, axis = 1
        ori axis with C, axis = 1 (NC1HWC0) 2(NDC1HWC0)
        ori axis with H, axis = 2 (NC1HWC0) 3(NDC1HWC0)
        ori axis with W, axis = 3 (NC1HWC0) 4(NDC1HWC0)
        '''
        offset_6hd = 1 if input_format == "NDC1HWC0" else 0
        format_c_axis = 1 + offset_6hd if not reduce_mode else [1 + offset_6hd, 4 + offset_6hd]
        format_axis_map = {
            "N": 0,
            "C": format_c_axis,
            "H": 2 + offset_6hd,
            "W": 3 + offset_6hd,
            "D": 1
        }
        concat_dim_name = ori_format[axis]
        axis = format_axis_map.get(concat_dim_name)

    if input_format in ("FRACTAL_NZ",):
        axis = axis % len(ori_shape)
        # when FRACTAL_NZ, mean: [A, B, C, D] -> [A, B, ceil(D//16), ceil(C//16), 16, 16]
        # update axis as follow:
        # ex: ori axis with last one dim, axis = the dim of ceil(D//16)
        # ex: ori axis with last second dim, axis = the dim of ceil(C//16)
        if axis == len(ori_shape) - 1:
            axis = len(ori_shape) - 2 if not reduce_mode else [len(ori_shape) - 2, len(ori_shape) + 1]
        elif axis == len(ori_shape) - 2:
            axis = len(ori_shape) - 1 if not reduce_mode else [len(ori_shape) - 1, len(ori_shape) + 0]

    if input_format in ("FRACTAL_Z", "FRACTAL_Z_3D"):
        axis = axis % len(ori_shape)
        # when FRACTAL_Z, mean: C1HWNiNoC0
        # when FRACTAL_Z_3D, mean: DC1HWNiNoC0
        offset_3d = 1 if input_format == "FRACTAL_Z_3D" else 0
        format_c_axis = 0 + offset_3d if not reduce_mode else [0 + offset_3d, 5 + offset_3d]
        format_n_axis = 3 + offset_3d if not reduce_mode else [3 + offset_3d, 4 + offset_3d]
        format_axis_map = {
            "N": format_n_axis,
            "C": format_c_axis,
            "H": 1 + offset_3d,
            "W": 2 + offset_3d,
            "D": 0
        }
        concat_dim_name = ori_format[axis]
        axis = format_axis_map.get(concat_dim_name)

    return axis


def nc1hwc0_to_nhwc(data: np.ndarray, ori_format: str, target_shape: Union[List, Tuple] = None, groups=None):
    """
    Convert the data format from NC1HWC0 to NHWC
    :param shape_from: the shape before convert
    :param shape_to: the shape after convert
    :param array: the one-dimensional array
    :return: the data array of NHWC shape
    """
    shape_from = data.shape
    n_from = shape_from[0]
    c1_from = shape_from[1]
    h_from = shape_from[2]
    w_from = shape_from[3]
    c0_from = shape_from[4]
    c1_mul_c0 = c1_from * c0_from
    c_pad = None if c1_mul_c0 == target_shape[-1] else target_shape[-1] - c1_mul_c0

    reshape_data = data.reshape(n_from, c1_from, h_from, w_from, c0_from)
    tmp_input_tensor = np.transpose(reshape_data, axes=(0, 2, 3, 1, 4))
    tmp_input_tensor = tmp_input_tensor.reshape((n_from, h_from, w_from, c1_from * c0_from))
    return tmp_input_tensor[:, :, :, :c_pad]


def nc1hwc0_to_nchw(data: np.ndarray, ori_format: str, target_shape: Union[List, Tuple] = None, groups=None):
    shape_from = data.shape
    n_from = shape_from[0]
    c1_from = shape_from[1]
    h_from = shape_from[2]
    w_from = shape_from[3]
    c0_from = shape_from[4]
    c1_mul_c0 = c1_from * c0_from
    c_pad = None if c1_mul_c0 == target_shape[1] else target_shape[1] - c1_mul_c0

    reshape_data = data.reshape(n_from, c1_from, h_from, w_from, c0_from)
    tmp_input_tensor = np.transpose(reshape_data, axes=(0, 1, 4, 2, 3))
    tmp_input_tensor = tmp_input_tensor.reshape((n_from, c1_from * c0_from, h_from, w_from))
    return tmp_input_tensor[:, :c_pad, :, :]


def nc1hwc0_to_hwcn(data: np.ndarray, ori_format: str, target_shape: Union[List, Tuple] = None, groups=None):
    shape_from = data.shape
    n_from = shape_from[0]
    c1_from = shape_from[1]
    h_from = shape_from[2]
    w_from = shape_from[3]
    c0_from = shape_from[4]
    c1_mul_c0 = c1_from * c0_from
    c_pad = None if c1_mul_c0 == target_shape[-2] else target_shape[-2] - c1_mul_c0

    reshape_data = data.reshape(n_from, c1_from, h_from, w_from, c0_from)
    tmp_input_tensor = np.transpose(reshape_data, axes=(2, 3, 1, 4, 0))
    tmp_input_tensor = tmp_input_tensor.reshape((h_from, w_from, c1_from * c0_from, n_from))
    return tmp_input_tensor[:, :, :c_pad, :]


def fractal_nz_to_nd(data: np.ndarray, ori_format: str, target_shape: Union[List, Tuple] = None, groups=None):
    shape_from = data.shape
    if len(target_shape) == 1:
        axis_h, axis_n, axis_c = 1, 1, target_shape[0]
    elif len(target_shape) == 2:
        axis_h, axis_n, axis_c = 1, target_shape[0], target_shape[1]
    else:
        axis_h, axis_n, axis_c = reduce(lambda x, y: x * y, target_shape[:-2]), target_shape[-2], target_shape[-1]
    axis_c1 = shape_from[-4]
    axis_no = shape_from[-3]
    axis_ni = shape_from[-2]
    axis_c0 = shape_from[-1]
    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
    n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
    tmp_input_tensor = data.reshape(axis_h, axis_c1, axis_no, axis_ni, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(0, 2, 3, 1, 4))
    tmp_input_tensor = tmp_input_tensor.reshape((axis_h, axis_no * axis_ni, axis_c1 * axis_c0))
    data_y = tmp_input_tensor[:, :n_pad, :c_pad]
    if len(target_shape) <= 2:
        data_y = data_y.reshape([data_y.shape[1], data_y.shape[2]])
    return data_y


def fractal_nz_to_nchw(data: np.ndarray, ori_format: str, target_shape: Union[List, Tuple] = None, groups=None):
    shape_from = data.shape
    if len(target_shape) == 1:
        axis_h, axis_n, axis_c = 1, 1, target_shape[0]
    elif len(target_shape) == 2:
        axis_h, axis_n, axis_c = 1, target_shape[0], target_shape[1]
    else:
        axis_h, axis_n, axis_c = reduce(lambda x, y: x * y, target_shape[:-2]), target_shape[-2], target_shape[-1]
    axis_c0 = shape_from[-1]
    axis_ni = shape_from[-2]
    axis_no = shape_from[-3]
    axis_c1 = shape_from[-4]
    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
    n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
    tmp_input_tensor = data.reshape(axis_h, axis_c1, axis_no, axis_ni, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(0, 2, 3, 1, 4))
    tmp_input_tensor = tmp_input_tensor.reshape((axis_h, axis_no * axis_ni, axis_c1 * axis_c0))
    data_y = tmp_input_tensor[:, :n_pad, :c_pad]
    if len(target_shape) <= 2:
        data_y = data_y.reshape([data_y.shape[1], data_y.shape[2]])
    return data_y


def fractal_nz_to_nhwc(data: np.ndarray, ori_format: str, target_shape: Union[List, Tuple] = None, groups=None):
    shape_from = data.shape
    if len(target_shape) == 1:
        axis_h, axis_n, axis_c = 1, 1, target_shape[0]
    elif len(target_shape) == 2:
        axis_h, axis_n, axis_c = 1, target_shape[0], target_shape[1]
    else:
        axis_h, axis_n, axis_c = reduce(lambda x, y: x * y, target_shape[:-2]), target_shape[-2], target_shape[-1]
    axis_ni = shape_from[-2]
    axis_no = shape_from[-3]
    axis_c0 = shape_from[-1]
    axis_c1 = shape_from[-4]
    n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
    tmp_input_tensor = data.reshape(axis_h, axis_c1, axis_no, axis_ni, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(0, 2, 3, 1, 4))
    tmp_input_tensor = tmp_input_tensor.reshape((axis_h, axis_no * axis_ni, axis_c1 * axis_c0))
    data_y = tmp_input_tensor[:, :n_pad, :c_pad]
    if len(target_shape) <= 2:
        data_y = data_y.reshape([data_y.shape[1], data_y.shape[2]])
    return data_y

    
def fractal_z_to_nchw(data: np.ndarray, ori_format: str, target_shape: Union[List, Tuple] = None, groups=None):
    shape_from = data.shape
    axis_c = target_shape[1]
    axis_n = target_shape[0]
    axis_no = shape_from[1]
    axis_ni = shape_from[2]
    axis_h = target_shape[2]
    axis_w = target_shape[3]
    axis_c1 = shape_from[0] // (axis_h * axis_w)
    axis_c0 = shape_from[3]
    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
    n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
    tmp_input_tensor = data.reshape(axis_c1, axis_h, axis_w, axis_no, axis_ni, axis_c0)
    # transpose the shape from (c1,h,w,no,ni,c0) to (no,ni,c1,c0,h,w)
    tmp_input_tensor = np.transpose(tmp_input_tensor, (3, 4, 0, 5, 1, 2))
    tmp_input_tensor = tmp_input_tensor.reshape((axis_no * axis_ni, axis_c1 * axis_c0, axis_h, axis_w))
    return tmp_input_tensor[:n_pad, :c_pad, :, :, ]


def fractal_z_to_hwcn(data: np.ndarray, ori_format: str, target_shape: Union[List, Tuple] = None, groups=None):
    shape_from = data.shape
    axis_c = target_shape[2]
    axis_n = target_shape[3]
    axis_no = shape_from[1]
    axis_ni = shape_from[2]
    axis_h = target_shape[0]
    axis_w = target_shape[1]
    axis_c1 = shape_from[0] // (axis_h * axis_w)
    axis_c0 = shape_from[3]
    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
    n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
    tmp_input_tensor = data.reshape(axis_c1, axis_h, axis_w, axis_no, axis_ni, axis_c0)

    tmp_input_tensor = np.transpose(tmp_input_tensor, (1, 2, 0, 5, 3, 4))
    tmp_input_tensor = tmp_input_tensor.reshape((axis_h, axis_w, axis_c1 * axis_c0, axis_no * axis_ni))
    return tmp_input_tensor[:, :, :c_pad, :n_pad]


format_transformation_map = {
    "NHWC": {
        "NC1HWC0": to_nc1hwc0,
        "FRACTAL_Z": to_fractal_z,
        "FRACTAL_Z_C04": to_fractal_z_c04
    },
    "NCHW": {
        "NC1HWC0": to_nc1hwc0,
        "FRACTAL_Z": to_fractal_z,
        "FRACTAL_Z_C04": to_fractal_z_c04
    },
    "HWCN": {
        "NC1HWC0": to_nc1hwc0,
        "FRACTAL_Z": to_fractal_z,
        "FRACTAL_Z_C04": to_fractal_z_c04
    },
    "NDHWC": {
        "NDC1HWC0": to_ndc1hwc0,
        "FRACTAL_Z_3D": to_fractal_z_3d
    },
    "NCDHW": {
        "NDC1HWC0": to_ndc1hwc0,
        "FRACTAL_Z_3D": to_fractal_z_3d
    },
    "DHWCN": {
        "NC1HWC0": to_ndc1hwc0,
        "FRACTAL_Z_3D": to_fractal_z_3d
    },
    "ND": {
        "FRACTAL_NZ": nd_to_fractal_nz,
        "FRACTAL_Z": nd_to_fractal_z,
        "FRACTAL_ZN_RNN": nd_to_fractal_z
    },
    "NC1HWC0": {
        "NHWC": nc1hwc0_to_nhwc,
        "NCHW": nc1hwc0_to_nchw,
        "HWCN": nc1hwc0_to_hwcn

    },
    "FRACTAL_NZ": {
        "ND": fractal_nz_to_nd,
        "NCHW": fractal_nz_to_nchw,
        "NHWC": fractal_nz_to_nhwc
    },
    "FRACTAL_Z": {
        "NCHW": fractal_z_to_nchw,
        "HWCN": fractal_z_to_hwcn
    }
}


def is_transformable(ori_format, target_format):
    if ori_format in format_transformation_map:
        if target_format in format_transformation_map[ori_format]:
            return True
    return False


def transform(data, ori_format, target_format, target_shape: Union[List, Tuple] = None, groups=None):
    if is_transformable(ori_format, target_format):
        return format_transformation_map[ori_format][target_format](data, ori_format, target_shape, groups)
    return None