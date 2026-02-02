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

NAME = "name"
ABBR_NAME = "abbr_name"
DESC = "desc"
IS_REQUIRED = "is_required"

aoe_args = [
    {
        NAME: '--model',
        ABBR_NAME: '-m',
        DESC: 'Required. Model file',
        IS_REQUIRED: True,
    },
    {
        NAME: '--weight',
        ABBR_NAME: '-w',
        DESC: 'Weight file. Required when framework is Caffe',
    },
    {
        NAME: "--framework",
        ABBR_NAME: '-f',
        DESC: "Framework type (Caffe: 0, MindSpore: 1, Tensorflow: 3, Onnx: 5).\n"
              "Framework will be automatically identified by the suffix of the model file if not set.",
    },
    {
        NAME: '--model_path',
        DESC: "Model file path.",
    },
    {
        NAME: '--singleop',
        DESC: "Single op definition file."
    },
    {
        NAME: '--input_format',
        DESC: 'Format of input data. E.g.: "NCHW"'
    },
    {
        NAME: '--input_shape',
        DESC: "Shape of input data. Separate multiple nodes with semicolons (;).\n"
              "Use double quotation marks (\") to enclose each argument.\n"
              "E.g.: \"input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2\"",
    },
    {
        NAME: '--dynamic_batch_size',
        DESC: 'Set dynamic batch size. E.g.: "batchsize1,batchsize2,batchsize3"',
    },
    {
        NAME: '--dynamic_image_size',
        DESC: "Set dynamic image size. Separate multiple nodes with semicolons (;). "
              "Use double quotation marks (\") to enclose each argument.\n"
              "E.g.: \"imagesize1_height,imagesize1_width;imagesize2_height,imagesize2_width\"",
    },
    {
        NAME: '--dynamic_dims',
        DESC: "Set dynamic dims. Separate multiple nodes with semicolons (;). "
              "Use double quotation marks (\") to enclose each argument.\n"
              "E.g.: \"dims1_n1,dims1_n2;dims2_n1,dims2_n2\"",
    },
    {
        NAME: '--reload',
        DESC: "Recover from interrupt",
    },
    {
        NAME: '--job_type',
        ABBR_NAME: '-j',
        DESC: "Required. Job type (sgat: 1, opat: 2, )",
        IS_REQUIRED: True,
    },
    {
        NAME: '--progress_bar',
        DESC: "Display progress or not. on(default): dispaly progress, off: not display progress",
    },
    {
        NAME: '--device',
        DESC: "Specify device by id",
    },
    {
        NAME: '--tune_optimization_level',
        DESC: "Tune optimization level. The value range is [O1, O2]. "
              "O1 means deeper optimization, O2 means default optimization.",
    },
    {
        NAME: '--output',
        DESC: "Output file path&name(needn't suffix, will add .om automatically).",
    },
    {
        NAME: '--output_type',
        DESC: "Set net output type. Support FP32, FP16, UINT8, INT8. "
              "E.g.: FP16, indicates that all out nodes are set to FP16.\n "
              "\"node1:0:FP16;node2:1:FP32\", indicates setting the datatype of multiple out nodes.",
    },
    {
        NAME: '--aicore_num',
        DESC: "Set aicore num",
    },
    {
        NAME: '--virtual_type',
        DESC: "This parameter is only supported on Ascend 310P-Series."
              "Enable virtualization. 0(default): disable; 1: enable",
    },
    {
        NAME: '--out_nodes',
        DESC: "Output nodes designated by users. Separate multiple nodes with semicolons (;). "
              "Use double quotation marks (\") to enclose each argument.\n "
              "E.g.: \"node_name1:0;node_name1:1;node_name2:0\"",
    },
    {
        NAME: '--input_fp16_nodes',
        DESC: "Input node datatype is fp16. Separate multiple nodes with semicolons (;). "
              "Use double quotation marks (\") to enclose each argument. E.g.: \"node_name1;node_name2\""
    },
    {
        NAME: '--insert_op_conf',
        DESC: "Config file to insert new op",
    },
    {
        NAME: '--op_name_map',
        DESC: "Custom op name mapping file. Note: A semicolon(;) cannot be included in each path, "
              "otherwise the resolved path will not match the expected one.",
    },
    {
        NAME: '--is_input_adjust_hw_layout',
        DESC: "Intput node datatype is fp16 and format is NC1HWC0, used with input_fp16_nodes. "
              "E.g.: \"true,true,false,true\"",
    },
    {
        NAME: '--is_output_adjust_hw_layout',
        DESC: "Net output node datatype is fp16 and format is NC1HWC0, used with out_nodes. "
              "E.g.: \"true,true,false,true\"",
    },
    {
        NAME: '--disable_reuse_memory',
        DESC: "The switch of reuse memory. Default value is : 0. 0 means reuse memory, 1 means do not reuse memory.",
    },
    {
        NAME: '--fusion_switch_file',
        DESC: 'Set fusion switch file path'
    },
    {
        NAME: '--enable_scope_fusion_passes',
        DESC: "Validate the non-general scope fusion passes, multiple names can be set and separated by ','. "
              "E.g.: ScopePass1,ScopePass2,...."
    },
    {
        NAME: '--enable_single_stream',
        DESC: 'Enable single stream. true: enable; false(default): disable',
    },
    {
        NAME: '--enable_small_channel',
        DESC: 'Set enable small channel. 0(default): disable; 1: enable',
    },
    {
        NAME: '--compress_weight_conf',
        DESC: 'Config file to compress weight.'
    },
    {
        NAME: '--buffer_optimize',
        DESC: 'Set buffer optimize. "l2_optimize" (default) or "l1_optimize". Set "off_optimize" to close.'
    },
    {
        NAME: '--compression_optimize_conf',
        DESC: 'Config file to compress optimize.'
    },
    {
        NAME: '--sparsity',
        DESC: 'Optional; enable structured sparse. 0(default): disable; 1: enable'
    },
    {
        NAME: '--Fnonhomo_split',
        DESC: 'Subgraph nonhomogenous split optimization.'
    },
    {
        NAME: '--precision_mode',
        DESC: 'Precision mode, support force_fp16(default), allow_mix_precision, allow_fp32_to_fp16, '
        'must_keep_origin_dtype.'
    },
    {
        NAME: '--op_select_implmode',
        DESC: 'Set op select implmode. Support high_precision, high_performance.default: high_performance.'
    },
    {
        NAME: '--optypelist_for_implmode',
        DESC: 'Appoint which op to select implmode, cooperated with op_select_implmode. '
              'Separate multiple nodes with commas (,). Use double quotation marks (") to enclose each argument. '
              'E.g.: "node_name1,node_name2".'
    },
    {
        NAME: '--op_precision_mode',
        DESC: 'Set the path of operator precision mode configuration file (.ini).'
    },
    {
        NAME: '--modify_mixlist',
        DESC: 'Set the path of operator mixed precision configuration file.'
    },
    {
        NAME: '--keep_dtype',
        DESC: 'Retains the precision of certain operators in inference scenarios, by using a configuration file.'
    },
    {
        NAME: '--customize_dtypes',
        DESC: 'Set the path of custom dtypes configuration file.'
    },
    {
        NAME: '--op_debug_level',
        DESC: "Debug enable for TBE operator building.\n 0 (default): Disable debug;\n"
              "1: Enable TBE pipe_all, and generate the operator CCE file and Python-CCE mapping file (.json);\n "
              "2: Enable TBE pipe_all, generate the operator CCE file and Python-CCE mapping file (.json), "
              "and enable the CCE compiler -O0-g.\n"
              "3: Disable debug, and keep generating kernel file (.o and .json).\n"
              "4: Disable debug, keep generation kernel file (.o and .json) and generate the operator CCE file (.cce) "
              "and the UB fusion computing description file (.json)"
    },
    {
        NAME: '--tune_ops_file',
        DESC: 'Specify some operators for tuning in the configuration file.'
    },
    {
        NAME: '--Fdeeper_opat',
        DESC: 'Operator deeper optimization.'
    },
    {
        NAME: '--Fop_format',
        DESC: 'Operator format tuning.'
    },
    {
        NAME: '--log',
        DESC: 'Generate log with level. Support debug, info, warning, error, null.'
    }
]
