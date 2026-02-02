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

from json import dump
import json
from pathlib import Path
import re
import torch.nn as nn

from msit_llm.transform.model_parser.kind import mlp, attention, convert, mname
from components.utils.file_open_check import ms_open
from msit_llm.common.utils import load_file_to_read_common_check
from components.utils.constants import TENSOR_MAX_SIZE


def has_child(module: nn.Module) -> bool:
    children = list(module.children())

    return len(children) > 0


def filter_dropout_module(module: nn.Module):
    ret = []
    children = list(module.named_children())

    for n, c in children:
        sub_children = list(c.children())
        if ((len(sub_children) > 0 and isinstance(sub_children[0], nn.Dropout))
                or isinstance(c, nn.Dropout)):
            continue
        ret.append([n, c])

    return ret


def find_duplicate(modules):
    reprs = [repr(item) for item in modules]

    count = 1
    block = reprs[0]

    for r in reprs[1:]:
        if r == block:
            count += 1

    return count, modules[0]


def process_layer(name, layer: nn.Module):
    ret = {"name": name}

    for child_name, child in layer.named_children():
        lowered_name = mname(child).lower()
        sub = filter_dropout_module(child)
        size = len(sub)

        if size > 0:
            if "mlp" in lowered_name:
                ret["mlp"] = mlp(child_name, sub)
            elif "attention" in lowered_name:
                ret["attention"] = attention(child_name, sub, size)
            else:
                continue
        else:
            if "input_layernorm" in ret:
                ret["post_attention_layernorm"] = convert(child_name, child)
            else:
                ret["input_layernorm"] = convert(child_name, child)

    return ret


def build_model_tree(module: nn.Module):
    if not isinstance(module, nn.Module):
        raise ValueError("input should be torch.nn.Module")

    def dfs(ret, name, cur):
        if isinstance(cur, nn.ModuleList):
            if len(cur) == 0:
                return
            repeat_count, layer = find_duplicate(cur)
            repeat_block = process_layer(name, layer)
            ret.append({
                "name": name,
                "kind": "Layers",
                "repeat_count": repeat_count,
                "repeat_block": repeat_block
            })
        elif has_child(cur):
            for n, c in cur.named_children():
                dfs(ret, n, c)
        elif isinstance(cur, nn.Dropout):
            pass
        else:
            ret.append(convert(name, cur))

    children = []
    dfs(children, '', module)
    return {"name": mname(module), "children": children}


def model_to_json(model: nn.Module, name: str):
    with ms_open(f"{name}.json", mode="w") as ff:
        dump(build_model_tree(model), ff)


def get_transformer_name(module):
    for n, m in module.named_children():
        if len(list(m.children())) > 0:
            # has child
            return n + '.'
    return ''


def get_weight_names(model):
    parsed_model = build_model_tree(model)

    # Add model names
    is_rope = any([kw in str(parsed_model) for kw in 'rotary Rotary rope Rope ROPE'.split()])
    dic = {
        'model_name': model.config.model_type,
        'model_prefix': get_transformer_name(model),
        'pe_type': 'ROPE' if is_rope else 'ALIBI',
    }
    
    # Add layer names
    for node in parsed_model.get('children', {}):
        if node.get('kind') == 'Embedding':
            # word_embedings
            dic['word_embeddings'] = node.get('name', 'word_embeddings')
        elif 'Norm' in node.get('kind'):
            if 'embedding' in node.get('name', ''):
                # word_embedding_layernorm
                dic['word_embeddings_layernorm'] = node.get('name', '')
                dic['word_embeddings_layernorm_bias'] = node.get('bias', False)
            else:
                # layernorm
                dic['layernorm'] = node.get('name', 'layernorm')
                dic['layernorm_bias'] = node.get('bias', False)
        elif node.get('kind') == 'Layers':
            # layers
            dic['layers_prefix'] = node.get('name', 'layers')
            block = node.get('repeat_block', {})
            dic['input_layernorm'] = block.get('input_layernorm', {}).get('name', 'input_layernorm')
            dic['post_attention_layernorm'] = block.get(
                'post_attention_layernorm', {}).get('name', 'post_attention_layernorm')

            # attention
            attention_dic = block.get('attention', {})
            if len(attention_dic):
                dic['self_attention'] = attention_dic.get('name', 'self_attention')
                dic['query_key_value'] = attention_dic.get('w', {}).get('name', 'query_key_value')
                dic['query_key_value_bias'] = attention_dic.get('w', {}).get('bias', False) or \
                    attention_dic.get('q', {}).get('bias', False)
                dic['qkv_sep'] = [
                    node.get('name', 'qkv') 
                    for node_name, node in attention_dic.items() 
                    if node_name in 'q k v kv'.split()
                    ]
                dic['o_proj'] = attention_dic.get('o', {}).get('name', 'o_proj')
                dic['o_proj_bias'] = attention_dic.get('o', {}).get('bias', False)

            # mlp
            mlp_dic = block.get('mlp', {})  
            if len(mlp_dic):
                dic['mlp'] = 'mlp'
                ff = mlp_dic.get('ff', {})
                if len(ff) == 3:
                    dic['gate_up_proj'] = 'gate_up_proj'
                    dic['mlp_sep'] = [ff[0].get('name', ''), ff[2].get('name', '')]
                    dic['down_proj'] = ff[1].get('name', '')
                    dic['mlp_bias'] = ff[1].get('bias', '')
                elif len(ff) == 2:
                    dic['gate_up_proj'] = ff[0].get('name', '')
                    dic['mlp_sep'] = []
                    dic['down_proj'] = ff[1].get('name', '')
                    dic['mlp_bias'] = ff[1].get('bias', '')
        elif node.get('kind') == 'Linear':
            dic['lmhead'] = node.get('name', 'lmhead')

    parsed_model['weight_names'] = dic
    return parsed_model


def regex_search(pattern_list, content):
    g = None
    for pattern in pattern_list:
        g = re.search(pattern, content)
        if g is not None:
            break
    if g is None:
        return ''
    else:
        return g.group(1)


def parse_input_max_count(content):
    pattern_list = [
        r'IN_TENSOR_Q_LEN_INDEX = ([0-9]+);',
        r'IN_TENSOR_COUNT = ([0-9]+);',
    ]
    res = regex_search(pattern_list, content)
    try:
        max_count = int(res)
    except Exception:
        max_count = -1
    return max_count


def parse_by_idx(content):
    pattern_list = [
        r'(InTensorId : int \{\n(\s{4}.{,512}\n)+\};)',
        r'(DecoderModelTensorIdx : uint32_t \{\n(\s{4}.{,512}\n)+\};)',
    ]
    input_str = regex_search(pattern_list, content)
    if input_str == '':
        return []
    res = []
    for line in input_str.split('\n')[1:-1]:
        # line example IN_TENSOR_INPUT = 0, // input
        ll = line.split('//')[0]
        ll = ll.split('=')[0]
        ll = ll.split(',')[0]
        ll = ll.strip()
        if ll != '':
            res.append(ll)
    return res


def parse_by_cand(content):
    pattern_list = [
        r'InTensorCandiadates = \{\n\s{8}\{"default", \{((\n\s{12}.{,512})+)\}\n\s{8}\},',
        r'InTensorCandidates = \{\n\s{8}\{"default", \{((\n\s{12}.{,512})+)\}\n\s{8}\},',
        r'TensorCandidates = \{\n\s{8}\{"default", \{((\n\s{12}.{,512})+)\}\n\s{8}\},',
    ]
    input_str = regex_search(pattern_list, content)
    if input_str == '':
        return []
    res = input_str.replace('"', '').split(',')
    res = [ll.strip() for ll in res]
    return res


def parse_input_names(content):
    res = parse_by_idx(content)
    if not res:
        res = parse_by_cand(content)

    max_count = parse_input_max_count(content)
    if max_count == -1:
        if 'IN_TENSOR_MAX' in res:
            max_count = res.index('IN_TENSOR_MAX') - 1
    if max_count > 0:        
        res = res[:max_count]
    return res


def get_input_names(files):
    content = '\n'.join([Path(fp).read_text() for fp in files])
    return parse_input_names(content)   


def get_atb_model_names(files):
    content = '\n'.join([Path(fp).read_text() for fp in files])
    pattern_list = [
        r'REGISTER_MODEL\((.{,512})\);',
    ]
    res_str = regex_search(pattern_list, content)
    if res_str == '':
        return 'DecoderModel'
    
    return '_'.join([ss.strip() for ss in res_str.split(',')])


def update_weight_prefix(parsed_model, source_path):
    # Read index.json
    weight_name_list = []
    for fp in Path(source_path).glob('*.index.json'):
        if weight_name_list:
            break

        fp = load_file_to_read_common_check(str(fp))
        try:
            with ms_open(fp, max_size=TENSOR_MAX_SIZE) as ff:
                dd = json.load(ff)            
            weight_name_list = list(dd['weight_map'].keys())
        except Exception:
            weight_name_list = []
    if not weight_name_list:
        return
    
    dic = parsed_model.get('weight_names', {})
    for key in ["layers_prefix", "model_prefix", "word_embeddings", "word_embeddings_layernorm", "layernorm", "lmhead"]:
        if key not in dic:
            continue
        cur_prefix = dic.get(key, "")
        if cur_prefix == "":
            continue
        for weight_name in weight_name_list:
            if (cur_prefix + '.') in weight_name:
                new_prefix = weight_name[:weight_name.index(cur_prefix + '.')] + cur_prefix
                dic[key] = new_prefix
                break
    parsed_model['weight_names'] = dic


def fix_parsed_model(parsed_model):
    dic = parsed_model.get('weight_names', {})
    model_type = dic.get('model_name', '')
    if model_type == 'bloom':
        dic['lmhead'] = dic.get('word_embeddings')
    elif model_type == 'qwen':
        dic['mlp_sep'] = ['w2', 'w1']
        dic['down_proj'] = 'c_proj'
    parsed_model['weight_names'] = dic
