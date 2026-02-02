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

atc_args = [
    {
        NAME: '--mode',
        DESC: "Run mode. 0(default): generate offline model; 1: convert model to JSON format; 3: only pre-check; "
              "5: convert ge dump txt file to JSON format; 6: display model info; 30: convert original "
              "graph to execute-om for nano(offline model)",
    },
    {
        NAME: '--model',
        DESC: 'Required. Model file',
        IS_REQUIRED: True,
    },
    {
        NAME: '--weight',
        DESC: 'Weight file. Required when framework is Caffe',
    },
    {
        NAME: '--om',
        DESC: 'The model file to be converted to json',
    },
    {
        NAME: "--framework",
        DESC: "Required. Framework type. 0:Caffe; 1:MindSpore; 3:Tensorflow; 5:Onnx",
        IS_REQUIRED: True,
    },
    {
        NAME: '--input_format',
        DESC: 'Format of input data. E.g.: "NCHW"',
    },
    {
        NAME: '--input_shape',
        DESC: "Shape of static input data or shape range of dynamic input. "
              "Separate multiple nodes with semicolons (;). "
              "Use double quotation marks (\") to enclose each argument. \n E.g.: \"input_name1:n1,c1,h1,w1; "
              "input_name2:n2,c2,h2,w2\" \n \"input_name1:n1~n2,c1,h1,w1;input_name2:n3~n4,c2,h2,w2\"",
    },
    {
        NAME: '--input_shape_range',
        DESC: "This option is deprecated and will be removed in future version, please use input_shape instead.\n "
              "Shape range of input data. Separate multiple nodes with semicolons (;). "
              "Use double quotation marks (\") to enclose each argument.",
    },
    {
        NAME: '--dynamic_batch_size',
        DESC: 'Set dynamic batch size. E.g.: "batchsize1,batchsize2,batchsize3"',
    },
    {
        NAME: '--dynamic_image_size',
        DESC: "Set dynamic image size. Separate multiple nodes with semicolons (;). "
              "Use double quotation marks (\") to enclose each argument.\n "
              "E.g.: \"imagesize1_height,imagesize1_width;imagesize2_height,imagesize2_width\"",
    },
    {
        NAME: '--dynamic_dims',
        DESC: "Set dynamic dims. Separate multiple nodes with semicolons (;). "
              "Use double quotation marks (\") to enclose each argument.\n "
              "E.g.: \"dims1_n1,dims1_n2;dims2_n1,dims2_n2\"",
    },
    {
        NAME: '--singleop',
        DESC: "Single op definition file. atc will generate offline model(s) for single op if --singleop is set.",
    },
    {
        NAME: '--output',
        DESC: "Required. Output file path&name(needn't suffix, will add .om/.exeom automatically).\n "
              "If --model is set to 30, an additional dbg file will be generated.\n "
              "If --singleop is set, this arg specifies the directory to "
              "which the single op offline model will be generated.",
        IS_REQUIRED: True
    },
    {
        NAME: '--output_type',
        DESC: "Set net output type. Support FP32, FP16, UINT8, INT8. "
              "E.g.: FP16, indicates that all out nodes are set to FP16.\n "
              "\"node1:0:FP16;node2:1:FP32\", indicates setting the datatype of multiple out nodes.",
    },
    {
        NAME: '--check_report',
        DESC: "The pre-checking report file. Default value is: \"check_result.json\"",
    },
    {
        NAME: '--json',
        DESC: "The output json file path&name which is converted from a model",
    },
    {
        NAME: '--soc_version',
        DESC: "Required. The soc version.",
        IS_REQUIRED: True
    },
    {
        NAME: '--virtual_type',
        DESC: "Set whether offline model can run on the virtual devices under compute capability allocation.\n "
              "0 (default) : Disable virtualization; 1 : Enable virtualization.",
    },
    {
        NAME: '--core_type',
        DESC: "Set core type AiCore or VectorCore. VectorCore: use vector core. Default value is: AiCore",
    },
    {
        NAME: '--aicore_num',
        DESC: "Set aicore num",
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
              "Use double quotation marks (\") to enclose each argument. E.g.: \"node_name1;node_name2\"",
    },
    {
        NAME: '--insert_op_conf',
        DESC: "Config file to insert new op",
    },
    {
        NAME: '--op_name_map',
        DESC: "Custom op name mapping file\n "
              "Note: A semicolon(;) cannot be included in each path, "
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
        DESC: "Set fusion switch file path",
    },
    {
        NAME: "--enable_scope_fusion_passes",
        DESC: "validate the non-general scope fusion passes, multiple names can be set and separated by ','. "
              "E.g.: ScopePass1,ScopePass2,...",
    },
    {
        NAME: "--enable_single_stream",
        DESC: "Enable single stream. true: enable; false(default): disable",
    },
    {
        NAME: "--enable_small_channel",
        DESC: "Set enable small channel. 0(default): disable; 1: enable",
    },
    {
        NAME: '--enable_compress_weight',
        DESC: "Enable compress weight. true: enable; false(default): disable",
    },
    {
        NAME: '--compress_weight_conf',
        DESC: "Config file to compress weight",
    },
    {
        NAME: '--compression_optimize_conf',
        DESC: "Config file to compress optimize",
    },
    {
        NAME: '--sparsity',
        DESC: "Optional; enable structured sparse. 0(default): disable; 1: enable",
    },
    {
        NAME: "--buffer_optimize",
        DESC: 'Set buffer optimize. Support "l2_optimize" (default), "l1_optimize", "off_optimize"',
    },
    {
        NAME: '--mdl_bank_path',
        DESC: 'Set the path of the custom repository generated after model tuning.',
    },
    {
        NAME: '--op_precision_mode',
        DESC: "Set the path of operator precision mode configuration file (.ini)",
    },
    {
        NAME: "--precision_mode",
        DESC: 'precision mode, support force_fp16(default), force_fp32, cube_fp16in_fp32out, '
              'allow_mix_precision, allow_fp32_to_fp16, must_keep_origin_dtype, allow_mix_precision_fp16, '
              'allow_mix_precision_bf16, allow_fp32_to_bf16.',
    },
    {
        NAME: "--modify_mixlist",
        DESC: 'Set the path of operator mixed precision configuration file.',
    },
    {
        NAME: '--keep_dtype',
        DESC: 'Retains the precision of certain operators in inference scenarios by using a configuration file.',
    },
    {
        NAME: '--customize_dtypes',
        DESC: "Set the path of custom dtypes configuration file.",
    },
    {
        NAME: '--op_bank_path',
        DESC: 'Set the path of the custom repository generated after operator tuning with AutoTune.',
    },
    {
        NAME: '--op_select_implmode',
        DESC: 'Set op select implmode. Support high_precision, high_performance, '
              'high_precision_for_all, high_performance_for_all. default: high_performance',
    },
    {
        NAME: '--optypelist_for_implmode',
        DESC: "Appoint which op to select implmode, cooperated with op_select_implmode.\n "
              "Separate multiple nodes with commas (,). "
              "Use double quotation marks (\") to enclose each argument. E.g.: \"node_name1,node_name2\"",
    },
    {
        NAME: '--op_debug_level',
        DESC: "Debug enable for TBE operator building.\n "
              "0 (default): Disable debug; 1: Enable TBE pipe_all, and "
              "generate the operator CCE file and Python-CCE mapping file (.json);\n "
              "2: Enable TBE pipe_all, generate the operator CCE file and Python-CCE mapping file (.json), "
              "and enable the CCE compiler -O0-g.\n "
              "3: Disable debug, and keep generating kernel file (.o and .json)\n "
              "4: Disable debug, keep generation kernel file (.o and .json) and generate the operator CCE file "
              "(.cce) and the UB fusion computing description file (.json)",
    },
    {
        NAME: '--save_original_model',
        DESC: "Control whether to output original model. E.g.: true: output original model",
    },
    {
        NAME: '--log',
        DESC: "Generate log with level. Support debug, info, warning, error, null",
    },
    {
        NAME: '--dump_mode',
        DESC: "The switch of dump json with shape, to be used with mode 1. 0(default): disable; 1: enable.",
    },
    {
        NAME: '--debug_dir',
        DESC: "Set the save path of operator compilation intermediate files.\n Default value: ./kernel_meta",
    },
    {
        NAME: '--status_check',
        DESC: "switch for op status check such as overflow.0: disable; 1: enable.",
    },
    {
        NAME: '--external_weight',
        DESC: "Convert const to file constant, and save weight in file. "
              "0 (default): save weight in om.  1: save weight in file.",
    },
    {
        NAME: '--op_compiler_cache_dir',
        DESC: "Set the save path of operator compilation cache files. Default value: $HOME/atc_data",
    },
    {
        NAME: '--op_compiler_cache_mode',
        DESC: "Set the operator compilation cache mode. Options are disable(default), "
              "enable and force(force to refresh the cache)",
    },
    {
        NAME: '--display_model_info',
        DESC: 'enable for display model info; 0(default): close display, 1: open display.',
    },
    {
        NAME: '--shape_generalized_build_mode',
        DESC: 'For selecting the mode of shape generalization when build graph.\n '
              'shape_generalized: Shape will be generalized during graph build\n '
              'shape_precise: Shape will not be generalized, use precise shape.',
    },
    {
        NAME: '--op_debug_config',
        DESC: 'Debug enable for Operator memory detection. Options are disable(default), '
              'enter as the configuration file path',
    },
    {
        NAME: '--atomic_clean_policy',
        DESC: "For selecting the atomic op clean memory policy. 0: centralized clean.  1: separate clean.\n",
    },
    {
        NAME: '--deterministic',
        DESC: "For deterministic calculation. 0 (default): deterministic off. 1: deterministic on.",
    },
    {
        NAME: '--host_env_os',
        DESC: "OS type of the target execution environment. \n "
              "The parameters that support setting are the OS types of the opp package\n "
              "Supported host env os as list:linux. default: linux",
    },
    {
        NAME: '--host_env_cpu',
        DESC: "CPU type of the target execution environment.\n "
              "The parameters that support setting are the CPU types of the opp package\n "
              "Supported host env cpu as list:\nsupport cpu: aarch64 x86_64 , respond to os: linux\n default: aarch64",
    }
]
