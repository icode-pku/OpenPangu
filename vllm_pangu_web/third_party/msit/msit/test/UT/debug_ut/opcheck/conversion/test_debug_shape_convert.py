# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
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
from unittest import TestCase
from unittest.mock import patch, MagicMock

import numpy as np

from components.debug.common import logger
from msit_opcheck.conversion.shape_convert import (
    BLOCK_SIZE,
    PAD_C0_MAPPING,
    align_factor,
    gen_axes_for_transpose,
    determine_c0,
    _calculate_group,
    is_nchw_like,
    is_ndchw_like,
    nd_shape2fhd_shape,
    nd_shape2nz_shape,
    fhd2nd,
    shd2nd,
    nd2fhd,
    nz2nd,
    nd2nz,
    to_fractal_z,
    to_fractal_z_c04,
    to_fractal_z_3d,
    to_nc1hwc0,
    to_ndc1hwc0,
    nd_to_fractal_nz,
    nd_to_fractal_z,
    update_axis_for_npu_inner_format,
    nc1hwc0_to_nhwc,
    nc1hwc0_to_nchw,
    nc1hwc0_to_hwcn,
    fractal_nz_to_nd,
    fractal_nz_to_nchw,
    fractal_nz_to_nhwc,
    fractal_z_to_nchw,
    fractal_z_to_hwcn,
    format_transformation_map,
    is_transformable,
    transform
)

from msit_opcheck.utils import align, ceil_div, lcm


class TestShapeConvert(TestCase):
    def test_constant(self):
        target_block_size = 16
        target_map = {
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
        self.assertEqual(BLOCK_SIZE, target_block_size)
        self.assertEqual(PAD_C0_MAPPING, target_map)

    def test_align_factor(self):
        target = 16
        ret = align_factor()
        self.assertEqual(ret, target)

        ret = align_factor('bfloat16')
        self.assertEqual(ret, target)

        target = 4
        ret = align_factor('int64')
        self.assertEqual(ret, target)

    def test_gen_axes_for_transpose(slef):
        offset = 3
        base = [4, 5, 6]
        target = [0, 1, 2, 7, 8, 9]
        slef.assertEqual(gen_axes_for_transpose(offset, base), target)

    def test_determine_c0(self):
        dtype = np.zeros((1,), dtype=np.float32).dtype
        target = 16
        ret = determine_c0(dtype)
        self.assertEqual(ret, target)

        target = 2
        ret = determine_c0(dtype, target_shape=(2, 2))
        self.assertEqual(ret, target)

    def test__calculate_group(self):
        cin = 16
        cout = 18
        groups = 4
        c0 = 2
        cin_ori, cout_ori = cin // groups, cout // groups
        mag_factor0 = lcm(cin_ori, c0) // cin_ori
        mag_factor1 = lcm(cout_ori, BLOCK_SIZE) // cout_ori
        mag_factor = min(lcm(mag_factor0, mag_factor1), groups)
        cin_g = align(mag_factor * cin_ori, c0)
        cout_g = align(mag_factor * cout_ori, BLOCK_SIZE)
        target = {
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
        with patch.object(logger, "debug") as mock_debug:
            ret = _calculate_group(cin, cout, groups, c0)
            mock_debug.assert_called_with('cin:%d, cout:%d, groups:%d, group_dict:' % (cin, cout, groups), ret)
            self.assertEqual(ret, target)

    def test_is_nchw_like(self):
        shape = (3, 3, 3, 3)
        format = 'NCHW'
        ret = is_nchw_like(shape, format)
        self.assertTrue(ret)

        shape = (3, 3, 3)
        ret = is_nchw_like(shape, format)
        self.assertFalse(ret)

        shape = (3, 3, 3, 3)
        format = 'NCHWH'
        ret = is_nchw_like(shape, format)
        self.assertFalse(ret)

        format = 'NCHH'
        ret = is_nchw_like(shape, format)
        self.assertFalse(ret)

    def test_is_ndchw_like(self):
        shape = (3, 3, 3, 3, 3)
        format = 'NDCHW'
        ret = is_ndchw_like(shape, format)
        self.assertTrue(ret)

        shape = (3, 3, 3)
        ret = is_ndchw_like(shape, format)
        self.assertFalse(ret)

        shape = (3, 3, 3, 3, 3)
        format = 'NCHW'
        ret = is_ndchw_like(shape, format)
        self.assertFalse(ret)

        format = 'NDCHH'
        ret = is_ndchw_like(shape, format)
        self.assertFalse(ret)

    def test_nd_shape2fhd_shape(self):
        nd_shape = (3, 3, 3)
        nd_format = "NCHW"
        with self.assertRaises(RuntimeError) as context:
            nd_shape2fhd_shape(nd_shape, nd_format)
            self.assertEqual(str(context.exception), f"shape: {nd_shape} of format {nd_format} is not NCHW-like.")

        nd_shape = (3, 3, 3, 3)
        c0 = determine_c0("float16", None)
        n, c = nd_shape[nd_format.index("N")], nd_shape[nd_format.index("C")]
        h, w = nd_shape[nd_format.index("H")], nd_shape[nd_format.index("W")]
        c1 = ceil_div(c, c0)
        target = (n, c1, h, w, c0)
        ret = nd_shape2fhd_shape(nd_shape, nd_format)
        self.assertEqual(ret, target)

    def test_nd_shape2nz_shape(self):
        nd_shape = (3, 3, 3, 3)
        m, n = nd_shape[-2:]
        m0 = 16
        n0 = determine_c0("float16", None)
        m1 = ceil_div(m, m0)
        n1 = ceil_div(n, n0)
        target = tuple(nd_shape[:-2] + (n1, m1, m0, n0))
        ret = nd_shape2nz_shape(nd_shape)
        self.assertEqual(ret, target)

    def test_fhd2nd(self):
        data = np.random.random((3, 3, 3, 2, 2))
        nd_shape = (3, 3, 3)
        nd_format = "NHWC"
        with self.assertRaises(RuntimeError) as context:
            fhd2nd(data, nd_shape, nd_format)
            self.assertEqual(str(context.exception), f"shape: {nd_shape} of format {nd_format} is not NCHW-like.")

        nd_shape = (3, 3, 3, 4)
        fhd_shape = data.shape
        pad = 1 + (nd_shape[nd_format.index("C")] - 1) % fhd_shape[-1]
        main_block = data[:, :fhd_shape[1] - 1, :, :, :]  # main block
        tail_block = data[:, fhd_shape[1] - 1, :, :, :pad]  # tail block
        # NC1HWC0 -> NHWC1C0
        main_block = main_block.transpose((0, 2, 3, 1, 4))
        # NHWC1C0 -> NHWC
        main_block = main_block.reshape(main_block.shape[:3] + (-1,))
        # concatenate
        target = np.concatenate((main_block, tail_block), axis=-1)
        ret = fhd2nd(data, nd_shape, nd_format)
        self.assertTrue((ret == target).all())

        nd_format = "NCHW"
        pad = 1 + (nd_shape[nd_format.index("C")] - 1) % fhd_shape[-1]
        main_block = data[:, :fhd_shape[1] - 1, :, :, :]  # main block
        tail_block = data[:, fhd_shape[1] - 1, :, :, :pad]  # tail block
        # NC1HWC0 -> NHWC1C0
        main_block = main_block.transpose((0, 2, 3, 1, 4))
        # NHWC1C0 -> NHWC
        main_block = main_block.reshape(main_block.shape[:3] + (-1,))
        # concatenate
        nhwc = np.concatenate((main_block, tail_block), axis=-1)
        target = nhwc.transpose(("NHWC".index(nd_format[0]), "NHWC".index(nd_format[1]),
                                 "NHWC".index(nd_format[2]), "NHWC".index(nd_format[3])))
        ret = fhd2nd(data, nd_shape, nd_format)
        self.assertTrue((ret == target).all())

    def test_shd2nd(self):
        data = np.random.random((3, 3, 3, 2, 2, 2))
        nd_shape = (3, 3, 3, 3)
        nd_format = "NDHWC"
        with self.assertRaises(RuntimeError) as context:
            shd2nd(data, nd_shape, nd_format)
            self.assertEqual(str(context.exception), f"shape: {nd_shape} of format {nd_format} is not NDCHW-like.")

        nd_shape = (3, 3, 3, 2, 4)
        shd_shape = data.shape
        pad = 1 + (nd_shape[nd_format.index("C")] - 1) % shd_shape[-1]
        main_block = data[:, :, :shd_shape[2] - 1, :, :, :]
        tail_block = data[:, :, shd_shape[2] - 1, :, :, :pad]
        # NDC1HWC0 -> NDHWC1C0
        main_block = main_block.transpose((0, 1, 3, 4, 2, 5))
        # NDHWC1C0 -> NDHWC
        main_block = main_block.reshape(main_block.shape[:4] + (-1,))
        # concatenate
        target = np.concatenate((main_block, tail_block), axis=-1)
        ret = shd2nd(data, nd_shape, nd_format)
        self.assertTrue((ret == target).all())

        nd_format = "NDCHW"
        pad = 1 + (nd_shape[nd_format.index("C")] - 1) % shd_shape[-1]
        main_block = data[:, :, :shd_shape[2] - 1, :, :, :]
        tail_block = data[:, :, shd_shape[2] - 1, :, :, :pad]
        # NDC1HWC0 -> NDHWC1C0
        main_block = main_block.transpose((0, 1, 3, 4, 2, 5))
        # NDHWC1C0 -> NDHWC
        main_block = main_block.reshape(main_block.shape[:4] + (-1,))
        # concatenate
        ndhwc = np.concatenate((main_block, tail_block), axis=-1)
        target = ndhwc.transpose(("NDHWC".index(nd_format[0]), "NDHWC".index(nd_format[1]),
                                  "NDHWC".index(nd_format[2]), "NDHWC".index(nd_format[3]),
                                  "NDHWC".index(nd_format[4])))
        ret = shd2nd(data, nd_shape, nd_format)
        self.assertTrue((ret == target).all())

    def test_nd2fhd(self):
        data = np.random.random((3, 2, 5))
        nd_shape = data.shape
        nd_format = "NCHW"
        with self.assertRaises(RuntimeError) as context:
            nd2fhd(data, nd_format)
            self.assertEqual(str(context.exception), f"shape: {nd_shape} of format {nd_format} is not NCHW-like.")

        data = np.random.random((3, 2, 5, 3))
        ori_data = data
        data = np.transpose(ori_data, axes=(nd_format.index("N"), nd_format.index("H"),
                                            nd_format.index("W"), nd_format.index("C")))
        nd_shape = data.shape
        c = nd_shape[-1]
        c0 = determine_c0(data.dtype.name, None)
        c1c0 = align(c, c0)
        if c1c0 > c:
            zero_block = np.zeros(nd_shape[:3] + (c1c0 - nd_shape[3],), dtype=data.dtype)
            fhd = np.concatenate((data, zero_block), axis=-1).reshape((nd_shape[0], nd_shape[1], nd_shape[2], -1, c0))
        else:
            fhd = data.reshape((nd_shape[0], nd_shape[1], nd_shape[2], -1, c0))
        target = np.transpose(fhd, axes=(0, 3, 1, 2, 4))
        ret = nd2fhd(ori_data, nd_format)
        self.assertTrue((ret == target).all())

        nd_format = "NHWC"
        data = np.random.random((3, 2, 5, 3))
        nd_shape = data.shape
        c = nd_shape[-1]
        c0 = determine_c0(data.dtype.name, None)
        c1c0 = align(c, c0)
        if c1c0 > c:
            zero_block = np.zeros(nd_shape[:3] + (c1c0 - nd_shape[3],), dtype=data.dtype)
            fhd = np.concatenate((data, zero_block), axis=-1).reshape((nd_shape[0], nd_shape[1], nd_shape[2], -1, c0))
        else:
            fhd = data.reshape((nd_shape[0], nd_shape[1], nd_shape[2], -1, c0))
        target = np.transpose(fhd, axes=(0, 3, 1, 2, 4))
        ret = nd2fhd(data, nd_format)
        self.assertTrue((ret == target).all())

    def test_nz2nd(self):
        # (A0, A1, A2, ..., An, N1, M1, M0, N0) -> (A, N1, M1, M0, N0) -> (A, M1, M0, N1, N0)
        data = np.random.random((1, 1, 1, 1))
        nd_shape = (1, 1, 1, 1)
        ori_nd_shape = copy.deepcopy(nd_shape)
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
        target = np.reshape(nd, ori_nd_shape)
        ret = nz2nd(data, nd_shape)
        self.assertTrue((ret == target).all())

        data = np.random.random((1, 1, 1, 1, 1))
        nd_shape = (1, 1, 1, 1)
        ori_nd_shape = copy.deepcopy(nd_shape)
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
        target = np.reshape(nd, ori_nd_shape)
        ret = nz2nd(data, ori_nd_shape)
        self.assertTrue((ret == target).all())

    def test_nd2nz(self):
        data = np.random.random((1, 1))
        ret = nd2nz(data)
        self.assertEqual(ret.shape, (1, 1, 16, 16))
        self.assertTrue((data.sum() == ret.sum()).all())

        data = np.random.random((1, 1, 1))
        ret = nd2nz(data)
        self.assertEqual(ret.shape, (1, 1, 1, 16, 16))
        self.assertTrue((data.sum() == ret.sum()).all())

    def test_to_fractal_z(self):
        data = np.random.random((1, 1, 1))
        ori_format = "NCHW"
        with self.assertRaises(RuntimeError) as context:
            to_fractal_z(data, ori_format)
            self.assertEqual(str(context.exception), f"shape: {data.shape} of format {ori_format} is not NCHW-like.")

        data = np.random.random((1, 1, 1, 1))
        ret = to_fractal_z(data, ori_format)
        self.assertEqual(ret.shape, (1, 1, 16, 16))
        self.assertTrue((data.sum() == ret.sum()).all())

    def test_to_fractal_z_c04(self):
        data = np.random.random((1, 1, 1, 1))
        ori_format = "NCHW"
        ret = to_fractal_z_c04(data, ori_format)
        self.assertEqual(ret.shape, (1, 1, 16, 16))
        self.assertTrue((data.sum() == ret.sum()).all())

    def test_to_fractal_z_3d(self):
        data = np.random.random((1, 1, 1, 1, 1))
        ori_format = "NDCHW"
        ret = to_fractal_z_3d(data, ori_format)
        self.assertEqual(ret.shape, (1, 1, 16, 16))
        self.assertTrue((data.sum() == ret.sum()).all())

    def test_to_nc1hwc0(self):
        data = np.random.random((1, 1, 1, 1, 1))
        ori_format = "NCHW"
        with self.assertRaises(RuntimeError) as context:
            to_nc1hwc0(data, ori_format)
            self.assertEqual(str(context.exception), "Please check original format and original shape: "
                             f"NC1HWC0 transformer doesn't support {len(data.shape)}D shape")

        data = np.random.random((1, 1, 1, 1))
        ori_format = "NCHW"
        ret = to_nc1hwc0(data, ori_format)
        self.assertEqual(ret.shape, (1, 1, 1, 1, 16))
        self.assertTrue((data.sum() == ret.sum()).all())

    def test_to_ndc1hwc0(self):
        data = np.random.random((1, 1, 1, 1, 1))
        ori_format = "NDCHW"
        ret = to_ndc1hwc0(data, ori_format)
        self.assertEqual(ret.shape, (1, 1, 1, 1, 1, 16))
        self.assertTrue((data.sum() == ret.sum()).all())

    def test_nd_to_fractal_nz(self):
        data = np.random.random((1,))
        ori_format = "NCHW"
        with self.assertRaises(ValueError) as context:
            nd_to_fractal_nz(data, ori_format)
            self.assertEqual(str(context.exception), "If you want to convert the ND format to the NZ format, "
                             "the shape dimension of the input ND format data must be greater than or equal to 2.")

        data = np.random.random((1, 1, 1, 1))
        ret = nd_to_fractal_nz(data, ori_format)
        self.assertEqual(ret.shape, (1, 1, 1, 1, 16, 16))
        self.assertTrue((data.sum() == ret.sum()).all())

    def test_nd_to_fractal_z(self):
        data = np.random.random((1, 1, 1))
        ori_format = "NCHW"
        with self.assertRaises(ValueError) as context:
            nd_to_fractal_z(data, ori_format)
            self.assertEqual(str(context.exception), "If you want to convert the ND format to the fractal_z format, "
                             "the shape dimension of the input ND format data must be equal to 4.")

        data = np.random.random((1, 1, 1, 1))
        ret = nd_to_fractal_z(data, ori_format)
        self.assertEqual(ret.shape, (1, 1, 1, 1, 16, 16))
        self.assertTrue((data.sum() == ret.sum()).all())

    def test_update_axis_for_npu_inner_format(self):
        ori_shape = (1, 1, 1, 1)
        axis = 3
        input_format = "NDC1HWC0"
        ori_format = 'NCHW'
        target = 4
        ret = update_axis_for_npu_inner_format(ori_shape, axis, input_format, ori_format)
        self.assertEqual(ret, target)

        ori_format = 'NC1HWC0'
        target = 3
        ret = update_axis_for_npu_inner_format(ori_shape, axis, input_format, ori_format)
        self.assertEqual(ret, target)

        ori_format = 'FRACTAL_NZ'
        target = 2
        ret = update_axis_for_npu_inner_format(ori_shape, axis, input_format, ori_format)
        self.assertEqual(ret, target)

        ori_format = 'FRACTAL_Z'
        target = 2
        ret = update_axis_for_npu_inner_format(ori_shape, axis, input_format, ori_format)
        self.assertEqual(ret, target)

        ori_format = 'FRACTAL_Z_3D'
        target = 2
        ret = update_axis_for_npu_inner_format(ori_shape, axis, input_format, ori_format)
        self.assertEqual(ret, target)

    def test_nc1hwc0_to_nhwc(self):
        data = np.random.random((3, 3, 3, 3, 3))
        ori_format = 'NC1HWC0'
        target_shape = (3, 3, 3, 3)
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
        target = tmp_input_tensor[:, :, :, :c_pad]
        ret = nc1hwc0_to_nhwc(data, ori_format, target_shape)
        self.assertTrue((ret == target).all())

    def test_nc1hwc0_to_nchw(self):
        data = np.random.random((3, 3, 3, 3, 3))
        ori_format = 'NC1HWC0'
        target_shape = (3, 3, 3, 3)
        shape_from = data.shape
        n_from = shape_from[0]
        c1_from = shape_from[1]
        h_from = shape_from[2]
        w_from = shape_from[3]
        c0_from = shape_from[4]
        c1_mul_c0 = c1_from * c0_from
        c_pad = None if c1_mul_c0 == target_shape[-1] else target_shape[-1] - c1_mul_c0

        reshape_data = data.reshape(n_from, c1_from, h_from, w_from, c0_from)
        tmp_input_tensor = np.transpose(reshape_data, axes=(0, 1, 4, 2, 3))
        tmp_input_tensor = tmp_input_tensor.reshape((n_from, c1_from * c0_from, h_from, w_from))
        target = tmp_input_tensor[:, :c_pad, :, :]
        ret = nc1hwc0_to_nchw(data, ori_format, target_shape)
        self.assertTrue((ret == target).all())

    def test_nc1hwc0_to_hwcn(self):
        data = np.random.random((3, 3, 3, 3, 3))
        ori_format = 'NC1HWC0'
        target_shape = (3, 3, 3, 3)
        shape_from = data.shape
        n_from = shape_from[0]
        c1_from = shape_from[1]
        h_from = shape_from[2]
        w_from = shape_from[3]
        c0_from = shape_from[4]
        c1_mul_c0 = c1_from * c0_from
        c_pad = None if c1_mul_c0 == target_shape[-1] else target_shape[-1] - c1_mul_c0

        reshape_data = data.reshape(n_from, c1_from, h_from, w_from, c0_from)
        tmp_input_tensor = np.transpose(reshape_data, axes=(2, 3, 1, 4, 0))
        tmp_input_tensor = tmp_input_tensor.reshape((h_from, w_from, c1_from * c0_from, n_from))
        target = tmp_input_tensor[:, :, :c_pad, :]
        ret = nc1hwc0_to_hwcn(data, ori_format, target_shape)
        self.assertTrue((ret == target).all())

    def test_fractal_nz_to_nd(self):
        data = np.random.random((1, 1, 1, 1, 1))
        target_shape = (1, 1, 1, 1)
        ret = fractal_nz_to_nd(data, None, target_shape, None)
        self.assertEqual(ret.shape, (1, 1, 1))
        self.assertTrue((data.sum() == ret.sum()).all())

    def test_fractal_nz_to_nchw(self):
        data = np.random.random((1, 1, 1, 1, 1))
        target_shape = (1, 1, 1, 1)
        ret = fractal_nz_to_nchw(data, None, target_shape, None)
        self.assertEqual(ret.shape, (1, 1, 1))
        self.assertTrue((data.sum() == ret.sum()).all())

    def test_fractal_nz_to_nhwc(self):
        data = np.random.random((1, 1, 1, 1, 1))
        target_shape = (1, 1, 1, 1)
        ret = fractal_nz_to_nhwc(data, None, target_shape, None)
        self.assertEqual(ret.shape, (1, 1, 1))
        self.assertTrue((data.sum() == ret.sum()).all())

    def test_fractal_z_to_nchw(self):
        data = np.random.random((1, 1, 1, 1, 1))
        target_shape = (1, 1, 1, 1)
        ret = fractal_z_to_nchw(data, None, target_shape, None)
        self.assertEqual(ret.shape, target_shape)
        self.assertTrue((data.sum() == ret.sum()).all())

    def test_fractal_z_to_hwcn(self):
        data = np.random.random((1, 1, 1, 1, 1))
        target_shape = (1, 1, 1, 1)
        ret = fractal_z_to_hwcn(data, None, target_shape, None)
        self.assertEqual(ret.shape, target_shape)
        self.assertTrue((data.sum() == ret.sum()).all())

    def test_format_transformation_map(self):
        target = {
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
        self.assertEqual(format_transformation_map, target)

    def test_is_transformable(self):
        ori_format = 'FRACTAL_Z'
        target_format = 'HWCN'
        self.assertTrue(is_transformable(ori_format, target_format))

        ori_format = 'FRACTAL_Z'
        target_format = 'HWCND'
        self.assertFalse(is_transformable(ori_format, target_format))

    def test_transform(self):
        global format_transformation_map

        data = np.random.random((1, 1, 1, 1))
        ori_format = 'FRACTAL_Z'
        target_format = 'HWCND'
        ret = transform(data, ori_format, target_format)
        self.assertIsNone(ret)

        ori_format = 'FRACTAL_Z'
        target_format = 'HWCN'
        mock_func = MagicMock()
        mock_func.return_value = 'array'
        format_transformation_map[ori_format][target_format] = mock_func
        ret = transform(data, ori_format, target_format)
        mock_func.assert_called_with(data, ori_format, None, None)
        self.assertEqual(ret, 'array')
        format_transformation_map[ori_format][target_format] = fractal_z_to_hwcn
