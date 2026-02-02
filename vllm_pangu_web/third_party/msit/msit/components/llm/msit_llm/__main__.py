# Copyright (c) 2023-2025 Huawei Technologies Co., Ltd.
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


import os
import subprocess

from components.utils.constants import TENSOR_MAX_SIZE, FileCheckConst
from components.utils.file_open_check import ms_open
from components.utils.file_utils import (
    check_and_get_real_path,
    check_path_type,
    check_if_valid_dir_pattern_path
)
from components.utils.parser import BaseCommand
from components.utils.security_check import is_enough_disk_space_left
from components.utils.util import filter_cmd
from msit_llm.common.log import logger, set_log_level, LOG_LEVELS
from msit_llm.common.utils import (
    str2bool, check_positive_integer, check_device_integer, safe_string,
    check_ids_string, check_number_list, check_process_integer, check_dump_time_integer,
    check_data_can_convert_to_int, check_specified_rank_id, check_l1_norm,
    load_file_to_read_common_check, check_cosine_similarity, check_kl_divergence,
    check_token_range, NAMEDTUPLE_PRECISION_METRIC, NAMEDTUPLE_PRECISION_MODE,
    get_rank_id_from_torchair_data
)
from msit_llm.dump.initial import init_dump_task, clear_dump_task
from msit_llm.errcheck.process import process_error_check
from msit_llm.bc_analyze import Analyzer

from msit_llm.badcase_analyze.bad_case_analyze import BadCaseAnalyzer


LOG_LEVELS_LOWER = [ii.lower() for ii in LOG_LEVELS.keys()]


class DumpCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            '--only-save-desc',
            '-sd',
            required=False,
            dest="save_desc",
            action='store_true',
            default=False,
            help='0 When save tensor, 1 When only save tensor description instead of tensor')

        parser.add_argument(
            '--save-operation-ids',
            '-ids',
            required=False,
            dest="ids",
            type=check_ids_string,
            default="",
            help='Save Tensor Ids')

        parser.add_argument(
            '--execute-range',
            '-er',
            required=False,
            dest="range",
            type=check_number_list,
            default="0,0",
            help='The range of saving tensor.Eg:0,10.And please ensure that the input length does not exceed 500.')

        parser.add_argument(
            '--save-operation-child',
            '-child',
            required=False,
            dest="child",
            type=str2bool,
            default=True,
            help='Dump all data of child operations if True, do nothing if False.')

        parser.add_argument(
            '--save-time',
            '-time',
            required=False,
            dest="time",
            type=check_dump_time_integer,
            default=3,
            help='0 when only need dump data before execution, '
                 '1 when only need dump data after execution, '
                 '2 dump both before and after data,'
                 '3 dump input tensors before execution and output tensors after execution.')

        parser.add_argument(
            '--operation-name',
            '-opname',
            required=False,
            dest="opname",
            type=safe_string,
            default=None,
            help='Operation names need to dump.')

        parser.add_argument(
            '--save-tiling',
            '-tiling',
            required=False,
            dest="tiling",
            action='store_true',
            default=False,
            help='Dump all data of child operations if True, do nothing if False')

        parser.add_argument(
            '--exec',
            dest="exec",
            required=True,
            type=safe_string,
            default='',
            help='Exec command to run acltransformer model inference.'
                 'E.g: --exec \"bash run.sh patches/models/modeling_xxx.py\" ')

        parser.add_argument(
            '--output',
            '-o',
            dest="output",
            required=False,
            default='./',
            help='Data output directory.E.g:--output /xx/xxxx/xx')

        parser.add_argument(
            '--save-tensor-part',
            '-stp',
            required=False,
            dest="save_tensor_part",
            type=check_positive_integer,
            default=2,
            help='0 when only need dump intensor, '
                 '1 when only need dump outtensor, 2 both.')

        parser.add_argument(
            '--type',
            dest="type",
            required=False,
            nargs='+',
            default=['tensor', 'model'],
            choices=['model', 'layer', 'op', 'kernel', 'tensor', 'cpu_profiling', 'onnx', 'stats'],
            help='dump type.')

        parser.add_argument(
            '--device-id',
            '-device',
            required=False,
            dest="device_id",
            type=check_device_integer,
            default=None,
            help='Specify a single device ID for dumping data, will skip other devices.')

        parser.add_argument(
            '-seed',
            required=False,
            dest="set_random_seed",
            type=check_data_can_convert_to_int,
            nargs='?',
            const=2024,
            default=None,
            help='set random seed, will ensure that the random results are consistent with each run.')

        parser.add_argument(
            '--enable-symlink',
            '-symlink',
            required=False,
            dest="enable_symlink",
            action='store_true',
            default=False,
            help='Enable symbolic links for duplicate files (saves disk space and runtime).')

        parser.add_argument(
            '--config-path',
            '-config',
            dest="config_path",
            required=False,
            help='Dump configuration file path.E.g:--config-path /xx/xxxx/xx.json')

        parser.add_argument("--log-level", "-l", default="info", choices=LOG_LEVELS_LOWER, help="specify log level.")

    def handle(self, args, **kwargs):
        if args.exec:
            set_log_level(args.log_level)
            logger.warning("Please ensure that your execution command is secure.")

            check_if_valid_dir_pattern_path(args.output)
            args.output = check_and_get_real_path(args.output, FileCheckConst.WRITE_ABLE, must_exist=False)
            if not is_enough_disk_space_left(args.output):
                raise OSError("Please make sure that the remaining disk space in the dump path is greater than 2 GB")

            if args.config_path:
                check_path_type(args.config_path, FileCheckConst.FILE)
                args.config_path = check_and_get_real_path(args.config_path, FileCheckConst.READ_ABLE, is_strict=True)

            init_dump_task(args)
            # 有的大模型推理任务启动后，输入对话时有提示符，使用subprocess拉起子进程无法显示提示符
            cmds = args.exec.split()
            cmds = filter_cmd(cmds)
            subprocess.run(cmds, shell=False)
            clear_dump_task(args)
            return


class CompareCommand(BaseCommand):
    def add_arguments(self, parser, **kwargs):
        parser.add_argument(
            '--golden-path',
            '-gp',
            dest="golden_path",
            required=True,
            help='Golden data path. It supports directory or file.')

        parser.add_argument(
            '--my-path',
            '-mp',
            dest="my_path",
            required=True,
            help='Compared data path. It supports directory or file.')

        parser.add_argument(
            '--cmp-level',
            '-cl',
            dest="cmp_level",
            required=False,
            default=None,
            choices=["layer", "module", "api", "logits"],
            help='Compare level. only enabled for atb.')

        parser.add_argument(
            '--output',
            '-o',
            dest="output",
            required=False,
            default='./',
            help='Data output directory.E.g:--output /xx/xxxx/xx')

        parser.add_argument(
            '--op-mapping-file',
            '-mf',
            dest="mapping_file",
            required=False,
            default='',
            help='Operation mapping file directory.E.g:--op-mapping-file /xx/xxxx/xx')

        parser.add_argument(
            '--custom-algorithms',
            '-alg',
            required=False,
            nargs='+',
            help='custom comparing algorithms in format "python_file_path.py:function". \
                  Should better be a standalong file, and function should in format like \
                  "def foo(golden_tensor, my_tensor): return float_value, string_message"')

        parser.add_argument("--log-level", "-l", default="info", choices=LOG_LEVELS_LOWER, help="specify log level.")

        parser.add_argument(
            '--weight',
            '-w',
            action='store_true',
            help='Compare float weights and dequant weights, if True, do nothing if False')

        parser.add_argument(
            '--stats',
            '-st',
            action='store_true',
            help='Compare statistics, If set, will execute compare_statistics function')

        parser.add_argument(
            '--rank',
            '-r',
            dest="rank_id",
            required=False,
            type=check_specified_rank_id,
            default=None,
            help='rank id. It supports integer or None.')

    def handle(self, args, **kwargs):
        set_log_level(args.log_level)

        args.golden_path = check_and_get_real_path(args.golden_path, FileCheckConst.READ_ABLE)
        args.my_path = check_and_get_real_path(args.my_path, FileCheckConst.READ_ABLE)
        check_if_valid_dir_pattern_path(args.output)
        args.output = check_and_get_real_path(args.output, FileCheckConst.WRITE_ABLE, must_exist=False)

        if args.mapping_file:
            check_path_type(args.output, FileCheckConst.DIR)
            args.mapping_file = check_and_get_real_path(args.mapping_file, FileCheckConst.READ_ABLE)

        from msit_llm.compare.torchair_acc_cmp import get_torchair_ge_graph_path

        # Adding custom comparing algorithms
        if args.custom_algorithms:
            from components.utils.cmp_algorithm import register_custom_compare_algorithm

            for custom_compare_algorithm in args.custom_algorithms:
                register_custom_compare_algorithm(custom_compare_algorithm)

        # accuracy comparing for different scenarios
        torchair_ge_graph_path = get_torchair_ge_graph_path(args.my_path)
        if args.weight:
            check_path_type(args.golden_path, FileCheckConst.DIR)
            check_path_type(args.my_path, FileCheckConst.DIR)
            from msit_llm.compare.cmp_weight import compare_weight
            compare_weight(args.golden_path, args.my_path, args.output)

        elif torchair_ge_graph_path is not None:
            from msit_llm.compare.torchair_acc_cmp import acc_compare
            rank_info_existed = False
            if os.path.isdir(args.my_path):
                for entry in os.listdir(args.my_path):
                    if (
                        os.path.isdir(os.path.join(args.my_path, entry)) and
                        get_rank_id_from_torchair_data(entry) != -1
                    ):
                        rank_info_existed = True
                        break
            if not rank_info_existed and args.rank_id is not None:
                logger.warning('The directory structure of torchair data is old, '
                               'the rank parameter will not take effect.')
            acc_compare(args.golden_path, args.my_path, args.output, args.rank_id, rank_info_existed)
        else:
            from msit_llm.compare.atb_acc_cmp import compare_file
            from msit_llm.compare.cmp_mgr import CompareMgr, ArgsParam
            if os.path.isfile(args.golden_path) and os.path.isfile(args.my_path):
                compare_file(os.path.abspath(args.golden_path), os.path.abspath(args.my_path))
            else:
                args_param = ArgsParam(args.stats, args.golden_path, args.my_path, args.output,
                                       args.cmp_level, args.log_level, args.mapping_file, args.weight)
                golden_path = os.path.abspath(args.golden_path)
                my_path = os.path.abspath(args.my_path)
                cmp_mgr_instance = CompareMgr(golden_path, os.path.abspath(my_path), args_param)
                if cmp_mgr_instance.is_parsed_cmp_path():
                    cmp_mgr_instance.compare(args.output)


class OpcheckCommand(BaseCommand):
    def add_arguments(self, parser, **kwargs):
        parser.add_argument(
            '--input',
            '-i',
            required=True,
            help='input directory.E.g:--input OUTPUT_DIR/msit_dump_TIMESTAMP/tensors/device_id_PID/TID/')

        parser.add_argument(
            '--output',
            '-o',
            required=False,
            default='./',
            help='Data output directory.E.g:--output /xx/xxx/xx')

        parser.add_argument(
            '--operation-ids',
            '-ids',
            required=False,
            type=check_ids_string,
            default="",
            help='Save Tensor Ids.E.g:-ids 24_1,2_3_5')

        parser.add_argument(
            '--operation-name',
            '-opname',
            required=False,
            type=safe_string,
            default=None,
            help='Operation names need to dump.E.g:-opname self,linear')

        parser.add_argument(
            '--precision-metric',
            '-metric',
            required=False,
            nargs='+',
            default=[],
            choices=NAMEDTUPLE_PRECISION_METRIC._fields,
            help='Output more results of other precision metrics.E.g:-metric abs kl cos_sim')

        parser.add_argument(
            '--device-id',
            '-device',
            required=False,
            type=check_device_integer,
            default=0,
            help='Spicifies the NPU device to bu used.E.g.:-device 1')

        parser.add_argument(
            '--atb-rerun',
            '-rerun',
            required=False,
            action='store_true',
            default=False,
            help='Rerun atb operations if True. Compare outputs in dump data if False')

        parser.add_argument(
            '--custom-algorithms',
            '-alg',
            required=False,
            nargs='+',
            help='custom comparing algorithms in format "python_file_path.py:function". \
                  Should better be a standalong file, and function should in format like \
                  "def foo(golden_tensor, my_tensor): return float_value, string_message"')

        parser.add_argument(
            '--precision-mode',
            '-pmode',
            required=False,
            default="keep_origin_dtype",
            choices=NAMEDTUPLE_PRECISION_MODE._fields,
            help='Specifies the precision mode to calculate golden output. Keep origin dtype or translate all \
                float tensors to torch.float16/torch.float32 before calculating and comparing.E.g.:-pmode force_fp32')

        parser.add_argument("-l", "--log-level", default="info", choices=LOG_LEVELS_LOWER, help="specify log level")

        parser.add_argument(
            '--jobs',
            '-j',
            required=False,
            dest="jobs",
            type=check_process_integer,
            default=1,
            help='Set the number of processes. The maximum number is 8. E.g.: -j 2'
        )

        parser.add_argument(
            '--optimization-identify',
            '-opt',
            required=False,
            action='store_true',
            default=False,
            help='Identify the impact of FA/PA operator optimization on the precision of the operator. \
                This parameter is used in combination with the --atb-rerun/-rerun parameter.')

    def handle(self, args, **kwargs):
        set_log_level(args.log_level)

        check_path_type(args.input, FileCheckConst.DIR)
        args.input = check_and_get_real_path(args.input, FileCheckConst.READ_ABLE)
        check_if_valid_dir_pattern_path(args.output)
        args.output = check_and_get_real_path(args.output, FileCheckConst.WRITE_ABLE, must_exist=False)

        # Adding custom comparing algorithms
        if args.custom_algorithms:
            from components.utils.cmp_algorithm import register_custom_compare_algorithm

            for custom_compare_algorithm in args.custom_algorithms:
                register_custom_compare_algorithm(custom_compare_algorithm)

        try:
            from msit_llm.opcheck.opchecker import OpChecker
        except ImportError as e:
            raise ImportError("Import msit_llm opchecker error") from e

        op = OpChecker()
        logger.info("===================Opcheck start====================")
        op.start_test(args)
        logger.info("===================Opcheck end====================")


class ErrCheck(BaseCommand):
    def add_arguments(self, parser, **kwargs) -> None:
        parser.add_argument(
            '--exec',
            dest="exec",
            required=True,
            type=safe_string,
            default='',
            help='Executable command that running acl-transformer model inference. '
                 'User is responsible for the safeness of the input command. '
                 "E.g. --exec 'bash run.sh patches/models/modeling_xxx.py'."
        )

        parser.add_argument(
            '--type',
            dest="type",
            nargs='+',  # one or more
            choices=['overflow'],
            default=['overflow'],
            help="Types that perform different error detection tasks. "
                 "Multiple arguments will trigger all the providing functionalities."
        )

        parser.add_argument(
            '--output',
            '-o',
            dest="output",
            required=False,
            default='',
            help="Directory that stores the error information. If not provided, current directory will be used."
        )

        parser.add_argument(
            '--exit',
            dest='exit',
            required=False,
            action='store_true',
            default=False,
            help="Flag determines whether to exit the program after detecting an error. Defaults to False."
        )

        parser.add_argument("-l", "--log-level", default="info", choices=LOG_LEVELS_LOWER, help="specify log level")

    def handle(self, args, **kwargs) -> None:
        set_log_level(args.log_level)
        logger.warning("Please ensure that your execution command is secure.")

        if not args.output:
            args.output = os.getcwd()
        check_if_valid_dir_pattern_path(args.output)
        args.output = check_and_get_real_path(args.output, FileCheckConst.WRITE_ABLE, must_exist=False)

        process_error_check(args)


class Transform(BaseCommand):
    def add_arguments(self, parser, **kwargs) -> None:
        scenarios_info = [
            "[float atb to quant atb model] directory containing both cpp and h file",
            "[float atb to quant atb model] a single cpp file, will use the h file with a same name",
            "[torch to float cpp atb model] directory containing config.json and py for building transformers model",
            "[torch to float python atb model] directory containing config.json and py for building transformers model",
        ]
        scenarios_info_str = "; ".join([f"{id}.{ii}" for id, ii in enumerate(scenarios_info, start=1)])

        parser.add_argument(
            "-s",
            "--source",
            required=True,
            help="source path, could be:" + scenarios_info_str,
        )

        parser.add_argument(
            "-atb",
            "--atb_model_path",
            default="",
            help="[torch to float atb model] ATB model directory containing .cpp and .h files."
        )

        parser.add_argument(
            "--enable-sparse",
            action='store_true',
            help="[float atb to quant atb model] Enable transforming to sparse-quant model"
        )
        parser.add_argument(
            "--to-python",
            "-py",
            action='store_true',
            help="[torch to float python atb model] Enable transforming to python atb model",
        )
        parser.add_argument(
            "--to-quant",
            "-quant",
            action='store_true',
            help="[torch to float python atb model] Enable transforming to python quant atb model",
        )
        parser.add_argument(
            "--quant-disable-names",
            type=safe_string,
            default=None,
            help="[torch to float python atb model] file or ',' separated string for layer names skipping quant. "
                 "Default None for 'lm_head'",
        )
        parser.add_argument(
            "-a",
            "--analyze",
            action="store_true",
            help="[float atb to atb model] Analysis tool to analyze the compatibility of model operator migration"
        )

        parser.add_argument("-l", "--log-level", default="info", choices=LOG_LEVELS_LOWER, help="specify log level")

    def handle(self, args, **kwargs) -> None:
        set_log_level(args.log_level)

        args.source = check_and_get_real_path(args.source, FileCheckConst.READ_ABLE,
                                              is_strict=True)

        from msit_llm.transform.utils import get_transform_scenario, SCENARIOS

        scenario = get_transform_scenario(args.source, to_python=args.to_python)
        logger.info(f"Current scenario: {scenario}")
        if scenario == SCENARIOS.torch_to_float_python_atb:
            from msit_llm.transform.torch_to_atb_python import transform

            quant_disable_names = ["lm_head"]
            if args.quant_disable_names is not None and os.path.isfile(args.quant_disable_names):
                args.quant_disable_names = check_and_get_real_path(args.quant_disable_names, FileCheckConst.READ_ABLE,
                                                                   is_strict=True)
                args.quant_disable_names = load_file_to_read_common_check(args.quant_disable_names)
                with ms_open(args.quant_disable_names, max_size=TENSOR_MAX_SIZE) as ff:
                    quant_disable_names = [ii.strip() for ii in ff.readlines()]
            elif args.quant_disable_names is not None:
                quant_disable_names = args.quant_disable_names.split(',')

            transform(source_path=args.source, to_quant=args.to_quant, quant_disable_names=quant_disable_names)
        elif scenario == SCENARIOS.float_atb_to_quant_atb:
            if os.path.isdir(args.source):
                for file in os.listdir(args.source):
                    if file.endswith(".cpp") or file.endswith(".h"):
                        check_and_get_real_path(file, FileCheckConst.READ_ABLE,
                                                is_strict=True)

            from msit_llm.transform.float_atb_to_quant_atb import transform_quant

            transform_quant.transform_quant(source_path=args.source, enable_sparse=args.enable_sparse)
        elif scenario == SCENARIOS.torch_to_float_atb:
            from msit_llm.transform.torch_to_float_atb import transform_float

            if args.analyze:
                transform_float.transform_report(source_path=args.source)
            else:
                args.atb_model_path = check_and_get_real_path(args.atb_model_path, FileCheckConst.READ_ABLE,
                                                              is_strict=True)
                if os.path.isdir(args.atb_model_path):
                    for file in os.listdir(args.atb_model_path):
                        if file.endswith(".cpp") or file.endswith(".h"):
                            check_and_get_real_path(file, FileCheckConst.READ_ABLE,
                                                    is_strict=True)
                transform_float.transform_float(source_path=args.source, atb_model_path=args.atb_model_path)

        else:
            message = f"Neither config.json + py or cpp found in {args.source}, not supported"
            logger.error(message)
            raise ValueError(message)


class BCAnalyze(BaseCommand):
    def add_arguments(self, parser, **kwargs) -> None:
        parser.add_argument(
            '--golden',
            '-g',
            dest="golden",
            required=True,
            help="Golden result to compare with. It must be a valid csv path")

        parser.add_argument(
            '--test',
            '-t',
            dest="test",
            required=True,
            help="Test result to compare with the golden. It must be a valid csv path")

        parser.add_argument("-l", "--log-level", default="info", choices=LOG_LEVELS_LOWER, help="specify log level")

    def handle(self, args, **kwargs) -> None:
        set_log_level(args.log_level)

        args.golden = check_and_get_real_path(args.golden, FileCheckConst.READ_ABLE,
                                              file_type=FileCheckConst.CSV_SUFFIX)
        args.test = check_and_get_real_path(args.test, FileCheckConst.READ_ABLE,
                                            file_type=FileCheckConst.CSV_SUFFIX)
        Analyzer.analyze(golden=args.golden, test=args.test)  # 后缀名判断在这里


class BadCaseAnalyze(BaseCommand):
    def add_arguments(self, parser, **kwargs) -> None:
        parser.add_argument(
            '--golden-path',
            '-gp',
            dest="golden_path",
            required=True,
            help="Golden result to compare with. It must be a valid csv path")

        parser.add_argument(
            '--my-path',
            '-mp',
            dest="my_path",
            required=True,
            help="My result to compare with the golden. It must be a valid csv path")

        parser.add_argument("-l", "--log-level", default="info", choices=LOG_LEVELS_LOWER, help="specify log level")

    def handle(self, args, **kwargs) -> None:
        set_log_level(args.log_level)

        args.golden_path = check_and_get_real_path(args.golden_path, FileCheckConst.READ_ABLE,
                                                   file_type=FileCheckConst.CSV_SUFFIX)
        args.my_path = check_and_get_real_path(args.my_path, FileCheckConst.READ_ABLE,
                                               file_type=FileCheckConst.CSV_SUFFIX)
        BadCaseAnalyzer.analyze(golden_csv_path=args.golden_path, test_csv_path=args.my_path)


class LogitsDump(BaseCommand):
    def add_arguments(self, parser, **kwargs) -> None:
        parser.add_argument(
            '--exec',
            '-e',
            dest="exec",
            required=True,
            type=safe_string,
            help="Test cmd for modeltest. It must be a valid cmd")

        parser.add_argument(
            '--bad-case-result',
            '-bcr',
            dest="bad_case_result_csv",
            required=True,
            help="Bad case result csv file from BadCaseAnalyze tool, It must be a valid csv path")

        parser.add_argument(
            '--token-range',
            '-tr',
            dest="token_range",
            type=check_token_range,
            default=1,
            help="Token range for logits dump, will dump '0~token_range-1' token's logits, default=1")

        parser.add_argument("--log-level", "-l", default="info", choices=LOG_LEVELS_LOWER, help="specify log level")

    def handle(self, args, **kwargs) -> None:
        set_log_level(args.log_level)
        logger.warning("Please ensure that your execution command is secure.")

        args.bad_case_result_csv = check_and_get_real_path(args.bad_case_result_csv, FileCheckConst.READ_ABLE,
                                                           FileCheckConst.CSV_SUFFIX)

        from msit_llm.logits_dump.logits_dump import LogitsDumper

        logits_dumper = LogitsDumper(args)
        logits_dumper.dump_logits()


class LogitsCompare(BaseCommand):
    def add_arguments(self, parser, **kwargs) -> None:
        parser.add_argument(
            '--golden-path',
            '-gp',
            dest="golden_path",
            required=True,
            help="Golden result to compare with. It must be a valid folder path")

        parser.add_argument(
            '--my-path',
            '-mp',
            dest="my_path",
            required=True,
            help="My result to compare with the golden. It must be a valid folder path")

        parser.add_argument(
            "--cosine-similarity",
            "-cs",
            dest="cosine_similarity",
            type=check_cosine_similarity,
            default=0.999,
            help="Metric value of cosine similarity, default 0.999"
        )

        parser.add_argument(
            "--kl-divergence",
            "-kl",
            dest="kl_divergence",
            type=check_kl_divergence,
            default=0.0001,
            help="Metric value of KL divergence, default 0.0001"
        )

        parser.add_argument(
            "--l1-norm",
            "-l1",
            dest="l1_norm",
            type=check_l1_norm,
            default=0.01,
            help="Metric value of L1_Norm, default 0.01"
        )

        parser.add_argument(
            "--dtype",
            "-d",
            dest="dtype",
            type=str,
            choices=['fp16', 'bf16', 'fp32'],
            default='fp16',
            help="The data precision types required for calculating ULP, default fp16"
        )

        parser.add_argument(
            '--output-dir',
            '-o',
            dest="output_dir",
            default='./output',
            help="Data output directory. E.g: '--output /xx/xxxx/xx', default=./output")

        parser.add_argument("--log-level", "-l", default="info", choices=LOG_LEVELS_LOWER, help="specify log level")

    def handle(self, args, **kwargs) -> None:
        set_log_level(args.log_level)

        check_path_type(args.golden_path, FileCheckConst.DIR)
        check_path_type(args.my_path, FileCheckConst.DIR)
        check_if_valid_dir_pattern_path(args.output_dir)
        args.golden_path = check_and_get_real_path(args.golden_path, FileCheckConst.READ_ABLE)
        args.my_path = check_and_get_real_path(args.my_path, FileCheckConst.READ_ABLE)
        args.output_dir = check_and_get_real_path(args.output_dir, FileCheckConst.WRITE_ABLE, must_exist=False)

        from msit_llm.logits_compare.logits_cmp import LogitsComparison

        logits_cmp = LogitsComparison(args)
        logits_cmp.process_comparsion()


def get_cmd_instance():
    llm_help_info = "Large Language Model(llm) Debugger Tools."
    dump_cmd_instance = DumpCommand("dump", "Dump tool for ascend transformer boost", alias_name="dd")
    compare_cmd_instance = CompareCommand("compare", "Accuracy compare tool for large language model", alias_name="cc")
    opcheck_cmd_instance = OpcheckCommand("opcheck", "Operation check tool for large language model", alias_name='oo')
    errcheck_cmd_instance = ErrCheck("errcheck", "Error check tool for large language model.", alias_name='ee')
    transform_cmd_instance = Transform("transform", "Transform tool for large language model.")
    bc_analyze_cmd_instance = BCAnalyze("analyze", "Bad Case analyze tool for large language model.")
    logits_bc_analyze_cmd_instance = BadCaseAnalyze('bcanalyze', "Bad case analyze tool for logits compare tool.")
    logits_dump_cmd_instance = LogitsDump('logitsdump', "Logits dump tool for logits compare tool.")
    logits_compare_cmd_instance = LogitsCompare('logitscmp', "Logits comparison tool for logits compare tool.")

    instances = [
        dump_cmd_instance, compare_cmd_instance, opcheck_cmd_instance, errcheck_cmd_instance, transform_cmd_instance,
        bc_analyze_cmd_instance, logits_bc_analyze_cmd_instance, logits_dump_cmd_instance, logits_compare_cmd_instance
    ]
    return BaseCommand("llm", llm_help_info, instances)
