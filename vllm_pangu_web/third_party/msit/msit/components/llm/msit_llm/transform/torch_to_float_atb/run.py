# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import os
from pathlib import Path
from typing import Optional
import importlib
import sys
from dataclasses import dataclass

import torch
from transformers.configuration_utils import PretrainedConfig

from msit_llm.common.utils import load_file_to_read_common_check, safe_int_env
from atb_llm.utils import file_utils, bind_cpus, initialize_distributed
from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log
from examples.server.cache import ModelConfig
from atb_llm.runner.model_runner import ModelRunner
from examples.run_pa import parse_arguments, PARunner
from components.utils.file_open_check import ms_open
from components.utils.constants import TENSOR_MAX_SIZE


@dataclass
class RouterParam:
    model_name_or_path: str
    max_position_embeddings: Optional[int] = None
    is_flash_causal_lm: bool = True
    load_tokenizer: bool = True
    revision: Optional[str] = None
    trust_remote_code: bool = False


def get_model(param):
    param.model_name_or_path = file_utils.standardize_path(param.model_name_or_path)
    file_utils.check_path_permission(param.model_name_or_path)
    model_type_key = 'model_type'
    config_dict, _ = PretrainedConfig.get_config_dict(param.model_name_or_path)
    config_dict[model_type_key] = config_dict[model_type_key].lower()
    model_type = config_dict[model_type_key]
    if model_type == "kclgpt":
        model_type = "codeshell"
    elif model_type == "internvl_chat":
        model_type = "internvl"
    if model_type == "qwen2_moe":
        model_type = model_type.replace('_', '')

    sys.path.append(f"{os.path.dirname(__file__)}/..")
    router_path = f"{model_type}.router_{model_type}"
    router = importlib.import_module(router_path)
    router_cls = getattr(router, f"{model_type.capitalize()}Router")
    router_ins = router_cls(
        param.model_name_or_path,
        param.max_position_embeddings,
        param.is_flash_causal_lm,
        param.load_tokenizer,
        param.revision,
        param.trust_remote_code,
        config_dict)
    return router_ins


class TransModelRunner(ModelRunner):
    def __init__(
        self, model_name_or_path, rank, world_size,
        npu_id=None,
        local_rank=None,
        max_position_embeddings=None,
        is_flash_causal_lm: bool = True,
        load_tokenizer: bool = True,
        **kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.rank = rank
        self.local_rank = local_rank if local_rank is not None else rank
        self.npu_id = npu_id if npu_id is not None else self.local_rank
        self.world_size = world_size
        self.inference_mode = kwargs.get("inference_mode", "")

        if ENV.bind_cpu:
            try:
                bind_cpus(world_size, self.npu_id, ratio=1.0)
            except RuntimeError as e:
                print_log(rank, logger.info, e)
            except Exception:
                print_log(rank, logger.info, 'Skip binding cpu.')

        param = RouterParam(
            model_name_or_path, 
            max_position_embeddings, 
            is_flash_causal_lm,
            load_tokenizer=load_tokenizer,
            revision=None, 
            trust_remote_code=False,
            )
        router_ins = get_model(param)

        self.model_cls = router_ins.model_cls
        self.config = router_ins.config
        self.tokenizer = router_ins.tokenizer
        self.input_builder = router_ins.input_builder
        self.postprocessor = router_ins.postprocessor

        self.dtype = self.config.torch_dtype
        self.quantize = self.config.quantize
        self.kv_quant = self.config.kv_quant
        self.kv_cache_dtype = torch.int8 if self.config.kv_quant is not None else self.dtype

        print_log(rank, logger.info, f'model_runner.quantize: {self.quantize}\n, '
                                     f'model_runner.kv_quant: {self.kv_quant}\n, '
                                     f'model_runner.dytpe: {self.dtype}')

        if self.kv_quant is not None and self.kv_quant not in ['C8']:
            raise NotImplementedError(
                f'unsupported type: {self.kv_quant}, 此类型从权重文件config.json中的`kv_quant`字段中获取；'
                f'若config.json中不存在此字段，请新增；当前此字段仅接受`C8`一种类型，'
                f'各模型具体支持的类型不同，请参考模型README文件。'
            )

        if self.dtype not in [torch.float16, torch.bfloat16]:
            raise NotImplementedError(
                f'unsupported type: {self.dtype}, 此类型从权重文件config.json中的`torch_dtype`字段中获取；'
                f'若config.json中不存在此字段，请新增；当前此字段仅接受`float16`和`bfloat16`两种类型，'
                f'各模型具体支持的类型不同，请参考模型README文件。'
            )

        self.lora_adapter = None
        lora_adapter_json_path = os.path.join(model_name_or_path, "lora_adapter.json")
        if os.path.exists(lora_adapter_json_path):
            lora_adapter_json_path = file_utils.standardize_path(lora_adapter_json_path)
            file_utils.check_file_safety(lora_adapter_json_path)
            with file_utils.safe_open(lora_adapter_json_path, mode="r", encoding="utf-8") as f:
                self.lora_adapter = json.load(f)

        self.process_group, self.device = initialize_distributed(self.rank, self.npu_id, world_size)
        torch.npu.set_compile_mode(jit_compile=False)

        print_log(rank, logger.info, f'init tokenizer done: {self.tokenizer}')


class TransPARunner(PARunner):
    def __init__(self, **kwargs):
        self.rank = kwargs.get('rank', '0')
        self.local_rank = kwargs.get('local_rank', self.rank)
        self.world_size = kwargs.get('world_size', '1')

        self.model_path = kwargs.get('model_path', None)
        self.lora_adapter = kwargs.get('lora_adapter', None)
        self.input_text = kwargs.get('input_text', None)

        self.max_batch_size = kwargs.get('max_batch_size', None)
        self.max_input_length = kwargs.get('max_input_length', None)
        self.max_output_length = kwargs.get('max_output_length', None)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', None)
        self.max_prefill_tokens = kwargs.get('max_prefill_tokens', None)

        self.block_size = kwargs.get('block_size', None)
        self.chat_template = kwargs.get('chat_template', None)
        self.is_flash_model = kwargs.get('is_flash_model', None)
        self.load_tokenizer = kwargs.get('load_tokenizer', True)

        self.model = TransModelRunner(
            self.model_path, rank=self.rank, world_size=self.world_size,
            local_rank=self.local_rank,
            max_position_embeddings=self.max_position_embeddings,
            load_tokenizer=self.load_tokenizer,
            lora_adapter=self.lora_adapter
        )
        self.tokenizer = self.model.tokenizer
        if self.chat_template:
            self.tokenizer.chat_template = self._load_chat_template(self.chat_template)
        self.dtype = self.model.dtype
        self.quantize = self.model.quantize
        self.kv_quant = self.model.kv_quant
        self.model.load_weights()

        self.device = self.model.device
        self.model_config = ModelConfig(self.model.num_heads,
                                        self.model.num_kv_heads,
                                        self.model.head_size,
                                        self.model.num_layers,
                                        self.model.device,
                                        self.model.dtype,
                                        self.model.soc_info,
                                        self.kv_quant)

        self.max_memory = NpuHbmInfo.get_hbm_capacity(self.local_rank, self.world_size, self.model.soc_info.need_nz)
        self.init_memory = int(
            self.max_memory * NpuHbmInfo.get_hbm_usage(self.local_rank, self.world_size, self.model.soc_info.need_nz))
        print_log(self.rank, logger.info, f'hbm_capacity(GB): {self.max_memory / (1024 ** 3)}, '
                                          f'init_memory(GB): {self.init_memory / (1024 ** 3)}')

        self.warm_up_memory = 0
        self.warm_up_num_blocks = 0
        self.cache_manager = None
        self.compress_head_enable = ENV.compress_head_enable


def main():
    args = parse_arguments()

    rank = safe_int_env("RANK", 0)
    local_rank = safe_int_env("LOCAL_RANK", 0)
    world_size = safe_int_env("WORLD_SIZE", 1)
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        **vars(args)
    }

    if args.input_ids:
        infer_inputs = args.input_ids
    else:
        infer_inputs = args.input_texts
    if args.is_chat_model and args.input_file:
        infer_inputs = []
        args.input_file = load_file_to_read_common_check(args.input_file)

        with ms_open(args.input_file, 'r', max_size=TENSOR_MAX_SIZE, encoding='utf-8') as file:
            for line in file:
                data_line = json.loads(line)
                infer_inputs.append(data_line)

    pa_runner = TransPARunner(**input_dict)
    print_log(rank, logger.info, f'pa_runner: {pa_runner}')
    pa_runner.warm_up()

    infer_params = {
        "inputs": infer_inputs,
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
        "is_chat_model": args.is_chat_model
    }
    generate_texts, token_nums, _ = pa_runner.infer(**infer_params)

    length = len(infer_inputs)
    for i, generate_text in enumerate(generate_texts):
        if i < length:
            print_log(rank, logger.info, f'Question[{i}]: {infer_inputs[i]}')
        print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
        print_log(rank, logger.info, f'Generate[{i}] token num: {token_nums[i]}')


if __name__ == '__main__':
    main()
