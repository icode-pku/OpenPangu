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

import os

import json
import inspect
import math 
from copy import deepcopy
from pathlib import Path

import torch
import torch_npu

try:
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.utils.fx import symbolic_trace
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(f"transformers package not found!") from error

from msit_llm.transform.utils import load_atb_speed, NPUSocInfo
from msit_llm.common.log import logger
from msit_llm.transform.utils import write_file
from msit_llm.transform.torch_to_atb_python.env import CONFIG_ATTR_CANDIDATES, FIXED_INPUTS, \
    KV_CACHE_SURFFIX, BASIC_INPUT_NAMES, VALID_NAME_CHARS, FLOAT_DTYPES, MINDIE_ATB_MODEL, \
    GATE_UP_WEIGHT, DOWN_WEIGHT


def find_mindie_supported_model(config):
    model_type = get_config_attr(config, "model_type", default=config)
    num_local_experts = get_config_attr(config, "num_local_experts", default=config)
    if model_type + str(num_local_experts) in MINDIE_ATB_MODEL.keys():
        return MINDIE_ATB_MODEL[model_type + str(num_local_experts)]
    return None


def get_expert_weights(layer_id, num_experts, flag, weights):
    #  layer_id: 0/1/2/3...    flag: GATE_UP_WEIGHT_  /DOWN_WEIHGT
    name = ""
    for key, _ in weights.items():
        if "layers.0" in key and "experts.0.w1" in key:
            name = key
    res = []
    if flag == "gate_up_weight_" and name is not None:
        for i in range(num_experts):
            w1 = weights[name.replace("layers.0", "layers." + str(layer_id)) \
                        .replace("experts.0.w1", "experts." + str(i)+".w1")]
            w3 = weights[name.replace("layers.0", "layers." + str(layer_id)) \
                        .replace("experts.0.w1", "experts." + str(i)+".w3")]
            res.append(torch.cat([w1, w3], dim=0))
    elif flag == "down_weight_" and name is not None:
        for i in range(num_experts):
            res.append(weights[name.replace("layers.0", "layers." + str(layer_id)) \
                            .replace("experts.0.w1", "experts." + str(i)+".w2")])
    return res


def get_config_attr(config, attr, default=None):
    if attr not in CONFIG_ATTR_CANDIDATES:
        return getattr(config, attr, default)

    for sub_attr in CONFIG_ATTR_CANDIDATES[attr]:
        if hasattr(config, sub_attr):
            return getattr(config, sub_attr)
    return default


def build_transformers_model(source_path):
    try:
        config = AutoConfig.from_pretrained(source_path, local_files_only=True)
        llm_model_config = get_config_attr(config, "text_config", default=config)
        is_vl_model = llm_model_config is not config
        model = AutoModelForCausalLM.from_config(llm_model_config)
    except Exception as e:
        raise ValueError(f"build model from {source_path} failed, make sure it works within transformers") from e
    return model, llm_model_config, is_vl_model


def to_transformers_traced_module(model, input_names=BASIC_INPUT_NAMES, disable_check=True):

    return symbolic_trace(model, input_names=input_names, disable_check=disable_check)


def get_lambda_source_code(function):
    source_code = inspect.getsource(function).split("function=")[-1].split(", inputs=")[0].strip()
    return source_code[:-1] if source_code.endswith(",") else source_code


def get_valid_name(name):
    return "".join([ii for ii in name if ii in VALID_NAME_CHARS])


def generate_infer_file(output_file, source_path, is_vl_model=False):

    file_name = "run_vl.py" if is_vl_model else "run.py"
    infer_file = Path(output_file).with_name(file_name)
    contents_str = Path(__file__).with_name(file_name).read_text()
    contents_str = contents_str.replace("atb_model_placeholder", Path(output_file).stem)
    contents_str = contents_str.replace("model_path_placeholder", os.path.abspath(source_path))
    write_file(str(infer_file), contents_str)
    return infer_file


class Operation:
    def __init__(
        self, op_type, op_param=None, inputs=None, outputs=None, op_name="", function=None, is_weights_first=False
    ):
        self.op_type, self.op_name, self.function, self.is_weights_first = op_type, op_name, function, is_weights_first
        self.op_param, self.inputs, self.outputs = op_param or {}, inputs or [], outputs or []

    def __repr__(self):
        dd = self.to_json()
        basic_info_keys = ["op_name", "op_type"]
        info = f"op_name={self.op_name}, op_type={self.op_type}\n  "
        info += "\n  ".join([f"{kk}={vv}" for kk, vv in dd.items() if kk not in basic_info_keys])
        return info

    def to_dict(self):
        return dict(
            op_type=self.op_type,
            op_param=self.op_param,
            inputs=self.inputs,
            outputs=self.outputs,
            op_name=self.op_name,
            function=self.function,
            is_weights_first=self.is_weights_first,
        )

    def to_json(self):
        # Keep only valid attributes
        dd = dict(op_type=self.op_type, inputs=self.inputs, outputs=self.outputs, op_name=self.op_name)
        if self.function is not None:
            dd["function"] = get_lambda_source_code(self.function)
        else:
            dd["op_param"] = json.dumps(self.op_param)
        return dd

    def copy(self):
        return Operation(**deepcopy(self.to_dict()))


class ATBModelConfig:
    def __init__(
        self,
        vocab_size=1,
        num_attention_heads=1,
        num_key_value_heads=-1,
        head_dim=1,
        max_batch_size=1,
        max_seq_len=1024,
        rope_theta=1e4,
        **kwargs,
    ):
        self.vocab_size, self.num_attention_heads, self.head_dim = vocab_size, num_attention_heads, head_dim
        self.max_batch_size, self.max_seq_len, self.kwargs = max_batch_size, max_seq_len, kwargs
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads > 0 else num_attention_heads
        self.rope_theta = rope_theta
        self.topk, self.num_layeres, self.num_experts = 0, 0, 0
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)

    def __repr__(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        return dict(
            vocab_size=self.vocab_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            **self.kwargs,
        )


class ATBModel:
    def __init__(self, atb_model, atb_model_config=None, output_shape=None, dtype="float16"):
        load_atb_speed()
        atb_model_config = atb_model_config or {}
        if isinstance(atb_model_config, dict):
            self.atb_model_config = ATBModelConfig(**atb_model_config)
        elif not isinstance(atb_model_config, ATBModelConfig):
            raise ValueError("atb_model_config is neither a dict or an instance of ATBModelConfig")
        else:
            self.atb_model_config = atb_model_config

        self.atb_model = atb_model
        self.inputs = self.input_names = set(atb_model.input_names)
        self.outputs = self.output_names = atb_model.output_names
        self.model_outputs, self.weights, self.inv_freq_weight, self.attention_mask = {}, {}, None, None

        if output_shape is not None and not isinstance(output_shape, (dict, list, tuple)):
            raise ValueError("output_shape should be None or type in one of (dict, list, tuple)")
        elif isinstance(output_shape, dict):
            required, provided = set(self.output_names), set(output_shape.keys())
            if required != provided:
                raise ValueError(f"output_shape is a dict with keys {provided}, while required are {required}")
        elif isinstance(output_shape, (list, tuple)):
            required, provided = len(self.output_names), len(output_shape)
            if required != provided:
                raise ValueError(f"output_shape len {provided} not equal to required len {required}")
        self.output_shape = output_shape

        if dtype not in FLOAT_DTYPES:
            raise ValueError(f"dtype={dtype} not supported, valid ones are {list(FLOAT_DTYPES.keys())}")
        self.dtype = FLOAT_DTYPES.get(dtype)

        self.soc_info = NPUSocInfo()
        self.block = 128
        self.head_dim = getattr(atb_model, "head_dim", self.atb_model_config.head_dim)
        self.num_attention_heads = getattr(atb_model, "num_attention_heads", self.atb_model_config.num_attention_heads)
        self.num_key_value_heads = getattr(atb_model, "num_key_value_heads", self.atb_model_config.num_key_value_heads)
        self.vocab_size = getattr(atb_model, "vocab_size", self.atb_model_config.vocab_size)
        self.rope_theta = getattr(atb_model, "rope_theta", self.atb_model_config.rope_theta)
        self.topk = getattr(atb_model, "topk", self.atb_model_config.topk)  
        self.num_layers = getattr(atb_model, "num_layeres", self.atb_model_config.num_layeres)
        self.num_experts = getattr(atb_model, "num_experts", self.atb_model_config.num_experts)
        self.gate_up_weights = []
        self.down_weights = []
        self.num_blocks = math.ceil(
            (self.atb_model_config.max_seq_len + 20) / 128 * self.atb_model_config.max_batch_size
        )
        if self.soc_info.need_nz:
            self.cache_shape = [
                self.num_blocks,
                self.head_dim * self.num_key_value_heads // 16,
                self.block,
                16
            ]
        else:
            self.cache_shape = [
                self.atb_model_config.max_batch_size,
                self.atb_model_config.max_seq_len,
                self.num_key_value_heads,
                self.head_dim,
            ]
        
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.kv_cache_names = [ii for ii in self.input_names if ii.split(".")[-1] in KV_CACHE_SURFFIX]
        self.past_key_values = {}
        self.atb_model_has_set_weights = hasattr(self.atb_model, "set_weights")  # Supported after RC3

    def __call__(self, input_ids=None, position_ids=None, slots_mapping=None, **kwargs):
        return self.forward(input_ids, position_ids, slots_mapping, **kwargs)

    def init_kv_cache(self):
        self.past_key_values = {ii: torch.zeros(self.cache_shape).to(self.dtype).npu() for ii in self.kv_cache_names}
        if self.soc_info.need_nz:
            for _, value in self.past_key_values.items():
                value = torch_npu.npu_format_cast_(value, 29)
        self.weights.update(self.past_key_values)
        if self.atb_model_has_set_weights:
            self.atb_model.set_weights(self.past_key_values)

    def set_weights(self, weights):
        source_weights = set(weights.keys())
        valid_weights = source_weights & self.inputs
        missing_weights = self.inputs - source_weights - set(FIXED_INPUTS)
        unused_weights = source_weights - self.inputs

        for kk in valid_weights:
            cur_weight = weights[kk]
            dtype_str = str(cur_weight.dtype).split(".")[-1]
            if dtype_str in FLOAT_DTYPES and cur_weight.dtype != self.dtype:
                cur_weight = cur_weight.to(self.dtype)
            self.weights[kk] = cur_weight.npu()

        for weight_name in unused_weights:
            if "invfreq" in weight_name.split(".")[-1].replace("_", "").lower():
                self.inv_freq_weight = weights[weight_name].squeeze().float().npu()
                unused_weights.remove(weight_name)
                break

        need_cos_sin_table = FIXED_INPUTS.cos_table in self.inputs or FIXED_INPUTS.sin_table in self.inputs
        if need_cos_sin_table and self.inv_freq_weight is None:
            logger.info(f"Set inv_freq_weight by rope_theta={self.rope_theta}")
            self.inv_freq_weight = self._calc_inv_freq_by_rope_theta()

        if FIXED_INPUTS.attention_mask in self.inputs:
            mask_tensor = torch.ones([self.atb_model_config.max_seq_len, self.atb_model_config.max_seq_len])
            attention_mask = torch.where((1 - torch.tril(mask_tensor)).to(torch.bool), -torch.inf, 0)
            self.attention_mask = attention_mask.to(self.dtype).npu()

            if self.soc_info.need_nz and attention_mask is not None:
                self.attention_mask = self.transdata_operation.execute([self.attention_mask])[0]
        
        for i in range(self.num_layers):
            if GATE_UP_WEIGHT+str(i) in self.inputs:
                cur_weights = get_expert_weights(i, self.num_experts, GATE_UP_WEIGHT, weights)
                self.gate_up_weights.append(torch.stack([i.transpose(0, 1) for i in cur_weights], dim=0).npu())
            if DOWN_WEIGHT+str(i) in self.inputs:
                cur_weights = get_expert_weights(i, self.num_experts, DOWN_WEIGHT, weights)
                self.down_weights.append(torch.stack(cur_weights, dim=0).npu())

        if self.atb_model_has_set_weights:
            self.atb_model.set_weights(self.weights)  # ATB provided function, no need to pass weights again

        if len(self.kv_cache_names) > 0:
            self.init_kv_cache()
            missing_weights -= set(self.kv_cache_names)

        if len(unused_weights) > 0:
            logger.warning(f"unused weights: {unused_weights}")
        if len(missing_weights) > 0:
            logger.warning(f"missing weights: {missing_weights}")

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, slots_mapping=None, **kwargs):
        # Basic inputs
        model_inputs, batch_size, cur_pos, input_len = {}, 1, 1, 1
        if not self.atb_model_has_set_weights:
            model_inputs.update(self.weights)
        if input_ids is not None:
            batch_size = input_ids.shape[0] if input_ids.dim() == 2 else 1
            input_len = cur_pos = input_ids.shape[-1]
            model_inputs[FIXED_INPUTS.input_ids] = input_ids.npu()
            logger.debug(f"batch_size = {batch_size}, input_len = {input_len}")
        if inputs_embeds is not None:  # For VL model
            batch_size = inputs_embeds.shape[0] if inputs_embeds.dim() == 3 else 1
            input_len = cur_pos = inputs_embeds.shape[-2]
            model_inputs[FIXED_INPUTS.inputs_embeds] = inputs_embeds.npu()
            logger.debug(f"batch_size = {batch_size}, input_len = {input_len}")
        if position_ids is not None:
            cur_pos = (position_ids[0, -1] if position_ids.dim() == 2 else position_ids[-1]).cpu() + 1
            model_inputs[FIXED_INPUTS.position_ids] = position_ids.npu()
            logger.debug(f"cur_pos = {cur_pos}")

        # inputs interpreted from others, or with default values
        if FIXED_INPUTS.slots_mapping in self.inputs:
            if slots_mapping is None:
                slots_mapping = torch.zeros([batch_size * input_len], dtype=torch.int)
            model_inputs[FIXED_INPUTS.slots_mapping] = slots_mapping.npu()

        # Check kwargs
        model_inputs.update({kk: vv.npu() for kk, vv in kwargs.items()})
        if FIXED_INPUTS.cos_table in self.inputs or FIXED_INPUTS.sin_table in self.inputs:
            meets_cos_sin_table = FIXED_INPUTS.cos_table in model_inputs and FIXED_INPUTS.sin_table in model_inputs
            if not meets_cos_sin_table and self.inv_freq_weight is not None and position_ids is not None:
                # if either cos_table or sin_table is missing, the values of both will be calculated
                cos_table, sin_table = self._calc_cos_sin_table_from_inv_freq(position_ids)
                model_inputs[FIXED_INPUTS.cos_table], model_inputs[FIXED_INPUTS.sin_table] = cos_table, sin_table
        if FIXED_INPUTS.attention_mask in self.inputs and FIXED_INPUTS.attention_mask not in model_inputs:
            model_inputs[FIXED_INPUTS.attention_mask] = self.attention_mask
        if FIXED_INPUTS.seq_len in self.inputs and FIXED_INPUTS.seq_len not in model_inputs:
            seq_len = torch.ones([batch_size], dtype=torch.int) * cur_pos
            model_inputs[FIXED_INPUTS.seq_len] = seq_len.npu()

        # Show missing inputs, in some cases like testing scenario, this may not an error
        missing_inputs = self.inputs - set(model_inputs.keys()) - set(self.weights.keys())
        if len(missing_inputs) != 0:
            logger.warning(
                f"Missing inputs: {missing_inputs}\nProvided: {model_inputs.keys()}\nCall `set_weights` if not already"
            )  # Not raising error, model may still can be executed

        # Creats output. Here output_shape maybe None or a dict or list
        if self.output_shape is None:
            self.model_outputs = {ii: torch.ones([batch_size * input_len, self.vocab_size]).to(self.dtype).npu() \
                                  for ii in self.outputs}
        elif isinstance(self.output_shape, dict):
            self.model_outputs = {kk: torch.ones(vv).to(self.dtype).npu() for kk, vv in self.output_shape.items()}
        else:
            self.model_outputs = {kk: torch.ones(vv).to(self.dtype).npu() \
                                  for kk, vv in zip(self.outputs, self.output_shape)}

        # Run forward
        bind_map = {}
        if FIXED_INPUTS.seq_len in self.inputs:
            bind_map[FIXED_INPUTS.seq_len] = model_inputs[FIXED_INPUTS.seq_len].cpu()
        
        for i in range(self.num_layers):
            if GATE_UP_WEIGHT+str(i) in self.inputs:
                model_inputs[GATE_UP_WEIGHT+str(i)] = self.gate_up_weights[i]
            if DOWN_WEIGHT+str(i) in self.inputs:
                model_inputs[DOWN_WEIGHT+str(i)] = self.down_weights[i]

        return self.atb_model.forward(model_inputs, self.model_outputs, bind_map)

    def _calc_inv_freq_by_rope_theta(self):
        inv_freq_weight = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2) / self.head_dim))
        return inv_freq_weight.float().npu()

    def _calc_cos_sin_table_from_inv_freq(self, position_ids):
        logger.debug(f"self.inv_freq_weight.shape = {self.inv_freq_weight.shape}")
        logger.debug(f"position_ids.shape = {position_ids.shape}")
        left = self.inv_freq_weight[None, :, None]
        right = position_ids[:, None, :] if position_ids.dim() == 2 else position_ids[None, None, :]
        freq = (left.to(right.device).float() @ right.float()).transpose(1, 2)
        freq = torch.cat([freq, freq], dim=-1)[0].npu()  # has to be float npu values, and dim == 2
        logger.debug(f"freq.shape = {freq.shape}")
        return freq.cos().to(self.dtype), freq.sin().to(self.dtype)
