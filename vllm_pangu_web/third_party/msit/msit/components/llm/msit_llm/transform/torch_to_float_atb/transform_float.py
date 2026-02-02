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

from pathlib import Path
import os
import json
import csv
import torch
from msit_llm.common.log import logger
from msit_llm.transform.model_parser import kind, parser
from msit_llm.transform import torch_to_float_atb
from msit_llm.transform.torch_to_float_atb.utils import (get_repeat_box_layer, 
    dag_to_model, init_save_name, init_save_dir)
from msit_llm.transform.utils import write_file
from components.utils.file_open_check import ms_open


SMALL_NUM_CONFIG = 4


def try_setting_small_model(config):
    attr_list = [
        'num_hidden_layers',
        'num_layers',
        'n_layers',
        'kv_channels',
        'intermediate_size',
        'rotary_emb_base',
        'seq_length',
        'vocab_size',
    ]
    for attr in attr_list:
        config.__setattr__(attr, SMALL_NUM_CONFIG)
    return config
    

def build_model(source_path):
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("transformers package not found, try pip install transformers") from error

    try:
        config = AutoConfig.from_pretrained(source_path, local_files_only=True)
        config = try_setting_small_model(config)
        model = AutoModelForCausalLM.from_config(config)
    except Exception as error:
        raise ValueError(f"build model from {source_path} failed, make sure it works within transformers") from error
    return model


def transform_report(source_path, save_name=None, save_dir=None, is_repeat=True):
    # 大模型在cpu上，float32
    model = build_model(source_path).float().cpu()
    model_layers = get_repeat_box_layer(model)
    from ascend_utils.pytorch.dag.dag_torch_hook import DagTorchHook
    dag_node = DagTorchHook(model, torch.ones([1, 32]).long(), collapse_repeat_block=is_repeat)
    parsed_model_layers = dag_to_model(dag_node, is_repeat, model_layers) 
    parsed_model_layers = {"name": kind.mname(model), "children": parsed_model_layers}

    model_name_lower = parsed_model_layers.get("name", "model").lower()
    json_save_name = init_save_name(save_name if save_name else model_name_lower) + ".json"
    json_save_dir = init_save_dir(save_dir if save_dir else model_name_lower, sub_dir="")
    json_save_path = os.path.join(json_save_dir, json_save_name)
    with ms_open(json_save_path, mode="w") as ff:
        json.dump(parsed_model_layers, ff)
    logger.info(f"model info saved: {json_save_path}")

    headers = ["ori_op_name", "ori_op_type", "op_name", "op_type", "soc_type", "engine", "is_supported"]
    csv_save_name = init_save_name(save_name if save_name else model_name_lower) + "_operators.csv"
    csv_save_dir = init_save_dir(save_dir if save_dir else model_name_lower, sub_dir="")
    csv_file_path = os.path.join(csv_save_dir, csv_save_name)
    data = []
    ops_dict = {
        'relu': 'ActivationOperation', 
        'gelu': 'ActivationOperation', 
        'hardswish': 'ActivationOperation', 
        'LogSigmoid': 'ActivationOperation', 
        'as_strided': 'AsStridedOperation', 
        'all_gather': 'AllGatherOperation', 
        'all_reduce': 'AllReduceOperation', 
        'full': 'FillOperation', 
        'masked_fill': 'FillOperation', 
        'masked_fill_': 'FillOperation', 
        'transpose': 'TransposeOperation', 
        'cat': 'ConcatOperation', 
        'cumsum': 'CumsumOperation', 
        'matmul': 'MatmulOperation', 
        'gather': 'GatherOperation', 
        'LayerNorm': 'LayerNormOperation', 
        'RMSNorm': 'RmsNormOperation', 
        'Linear': 'LinearOperation', 
        'DistributedDataParallel': 'LinearParallelOperation', 
        'multinomial': 'MultinomialOperation', 
        'index_select': 'SliceOperation', 
        'split': 'SplitOperation', 
        'chunk': 'SplitOperation', 
        'softmax': 'SoftmaxOperation', 
        'repeat': 'RepeatOperation', 
        'where': 'WhereOperation', 
        'broadcast': 'BroadcastOperation', 
        'topk': 'TopkToppSamplingOperation',  
        'Embedding': 'WordEmbedding', 
        'LlamaRMSNorm': 'RmsNormOperation'
    }

    for node in dag_node.dag_node_list:
        if "forward" in node.name:
            continue

        ori_op_name = f"{node.name_in_network}"
        ori_op_type = f"{node.op_type}"
        op_name = None
        op_type = ops_dict.get(ori_op_type, None)
        soc_type = 'Ascend910'
        engine = 'UNKNOWN' if op_type is None else 'AICORE'
        is_supported = 'FALSE' if op_type is None else 'TRUE'

        data.append((ori_op_name, ori_op_type, op_name, op_type, soc_type, engine, is_supported))

    with ms_open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for row in data:
            writer.writerow(row)
    logger.info(f"csv info saved: {csv_file_path}")


def transform_float_cpp(parsed_model, save_name=None, save_dir=None):
    model_cpp_file, _ = torch_to_float_atb.float_model_cpp_gen(parsed_model, save_name=save_name, save_dir=save_dir)
    model_h_file, _ = torch_to_float_atb.float_model_h_gen(parsed_model, save_name=save_name, save_dir=save_dir)
    layer_cpp_file, _ = torch_to_float_atb.float_layer_cpp_gen(parsed_model, save_name=save_name, save_dir=save_dir)
    layer_h_file, _ = torch_to_float_atb.float_layer_h_gen(parsed_model, save_name=save_name, save_dir=save_dir)
    result_files = [model_cpp_file, model_h_file, layer_cpp_file, layer_h_file]

    logger.info("Generated files: [\n    " + ",\n    ".join(result_files) + ",\n]")
    return result_files


def save_run_py(parsed_model, save_dir=None):
    rr = Path(__file__).with_name("run.py").read_text()
    weight_names = parsed_model.get('weight_names', {})
    model_name_lower = weight_names.get('model_name', 'model')
    save_dir = init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir=".")
    save_path = os.path.join(save_dir, "run.py")
    write_file(save_path, rr)
    return save_path, rr


def transform_float_py(parsed_model, save_name=None, save_dir=None):
    result_files = []
    routing_py_file, _ = torch_to_float_atb.router_py_gen(parsed_model, save_dir=save_dir)
    result_files.append(routing_py_file)

    modeling_py_file, _ = torch_to_float_atb.modeling_py_gen(parsed_model, save_dir=save_dir)
    result_files.append(modeling_py_file)
    
    flash_causal_py, _ = torch_to_float_atb.flash_causal_py_gen(parsed_model, save_dir=save_dir)
    result_files.append(flash_causal_py)

    run_py, _ = save_run_py(parsed_model, save_dir=None)
    result_files.append(run_py)

    logger.info("Generated files: [\n    " + ",\n    ".join(result_files) + ",\n]")
    return result_files


def save_json(dic, name, save_name=None, save_dir=None):
    json_save_name = init_save_name(save_name if save_name else name) + ".json"
    json_save_dir = init_save_dir(save_dir if save_dir else name, sub_dir="")
    json_save_path = os.path.join(json_save_dir, json_save_name)
    with ms_open(json_save_path, mode="w") as ff:
        json.dump(dic, ff, indent=4)
    logger.info(f"model info saved: {json_save_path}")

    
def check_atb_model_path(atb_model_path):
    if atb_model_path == '':
        return []
    if Path(atb_model_path).is_dir():
        atb_files = list(Path(atb_model_path).glob('*.h')) + list(Path(atb_model_path).glob('*.cpp'))
        if len(atb_files) != 2:
            raise FileNotFoundError(f"Couldn't parse files in {atb_model_path} automatically "
                                    "because there should be one .cpp file and one .h files."
                                    f"Found {len(atb_files)} files.")
        return atb_files
    elif Path(atb_model_path).is_file(): 
        atb_files = []
        fp = Path(atb_model_path)
        if fp.suffix in ['.cpp', '.h']:
            atb_files = [str(fp.with_suffix('.cpp')), str(fp.with_suffix('.h'))]
        for ff in atb_files:
            if not Path(ff).exists():
                raise FileNotFoundError(f"Couldn't parse files in {atb_model_path} automatically "
                                    "because there should be one .cpp file and one .h files."
                                    f"File {ff} not found.")
        return atb_files
    else:
        raise FileNotFoundError(f"Couldn't parse files in {atb_model_path} automatically "
                                    "because there should be one .cpp file and one .h files. "
                                    "Please check.")

                
def transform_float(source_path, save_name=None, save_dir=None, atb_model_path=''):
    atb_files = check_atb_model_path(atb_model_path)

    logger.info("Building model using transformers...")

    model = build_model(source_path)

    logger.info("Transforming to atb")

    parsed_model = parser.get_weight_names(model)

    result_files = []

    if atb_files == []:
        result_files += transform_float_cpp(parsed_model, save_name, save_dir)
        atb_files = result_files[:2]

    parsed_model.get('weight_names', {})['model_name_in_atb_framework'] = parser.get_atb_model_names(atb_files)
    parsed_model['acl_inputs_name'] = parser.get_input_names(atb_files)
    parser.update_weight_prefix(parsed_model, source_path)
    result_files += transform_float_py(parsed_model, save_name, save_dir)

    fp_name = parsed_model.get("weight_names", {}).get("model_name", "").lower()
    save_json(parsed_model, fp_name, save_name=None, save_dir=None)
    return result_files

