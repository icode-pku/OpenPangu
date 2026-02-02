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

from msit_llm.transform.float_atb_to_quant_atb.utils import print_spelling, print_update_info
from msit_llm.transform.float_atb_to_quant_atb.transform_quant import IN_BETA, IN_HOLDER, BIAS_SUFFIX, INDEX_SUFFIX

NORM_PARAM = "NORM"
LINEAR_PARAM = "LINEAR"
MLP_PARAM = "MLP"
ATTENTION_PARAM = "ATTENTION"


class TransformQuantCppLayerFunction:
    def __init__(self, contents, cursor, in_tensor_added, indent=4, enable_sparse=False):
        self.contents, self.cursor, self.in_tensor_added = contents, cursor, in_tensor_added
        self.enable_sparse = enable_sparse

        self.in_tensor_added_enums = in_tensor_added
        self.in_tensor_added_params = [ii.lower() for ii in in_tensor_added if not ii.endswith(INDEX_SUFFIX)]
        self.all_tokens = list(cursor.get_tokens())
        self.all_token_len = len(self.all_tokens)
        self.total_id = self.all_token_len - 4  # starts from 4 and ends on -4, avoiding index overflow
        self.indent_prefix = " " * indent
        self.cur_param_index, self.cur_intensor_enum_index = 0, 0
        self.updates = []

        self.atb_node_params, self.atb_param_nodes = self.get_atb_nodes_and_parameters()
        self.atb_param_types = self.get_atb_parameters_type()
        self.param_groups = self.groupby_param_type()  # {param_type: group}
        self.norm_nodes = self.get_all_norm_nodes()
        self.need_to_check_mlp_norm = self.is_mlp_norm_node_using_attn_norm_param()

        self.is_separate_qkv, self.separate_qkv_nodes = False, []
        for param, nodes in self.atb_param_nodes.items():
            param_group = self.param_groups.get(self.atb_param_types.get(param), None)
            if (param_group == LINEAR_PARAM) and (len(nodes) > 1):
                self.is_separate_qkv, self.separate_qkv_nodes = True, nodes
                break

    def __call__(self):
        cur_id = 4  # starts from 4 and ends on -4, avoiding index overflow
        self.cur_param_index, self.cur_intensor_enum_index = 0, 0
        temp_atb_param_nodes = self.atb_param_nodes.copy()
        norm_count, linear_count, is_mlp_norm, is_separate_qkv_linear = 0, 0, False, False

        while cur_id < self.total_id:
            cur_token = self.all_tokens[cur_id]
            cur_token_spelling = cur_token.spelling
            if self.need_to_check_mlp_norm:
                is_mlp_norm = self.is_mlp_norm_node(cur_id) and norm_count > 0
            if self.is_separate_qkv:
                is_separate_qkv_linear = self.is_separate_qkv_node_operation(cur_token_spelling, cur_id)

            if not is_mlp_norm and not is_separate_qkv_linear and cur_token_spelling not in self.param_groups:
                cur_id += 1
                continue

            if is_mlp_norm:
                param_name = self.atb_node_params.get(cur_token_spelling, 'Not exist')
            elif is_separate_qkv_linear:
                param_name = self.atb_node_params(cur_token_spelling, 'Not exist')
            else:
                param_name = self.all_tokens[cur_id + 1].spelling

            if param_name not in self.atb_param_types or param_name not in temp_atb_param_nodes:
                cur_id += 1
                continue
            node_name = temp_atb_param_nodes[param_name].pop(0)
            if len(temp_atb_param_nodes[param_name]) == 0:
                temp_atb_param_nodes.pop(param_name)
            print_spelling(self.all_tokens[cur_id: cur_id + 3], info="current 3 tokens: ")

            if self.enable_sparse and linear_count == 0 and cur_token_spelling == "LinearParam":
                self.update_for_sparse_linear_param(cur_id)

            if is_mlp_norm:
                norm_count += 1
                cur_id = self.update_for_mlp_norm(cur_id, param_name, node_name)
            elif is_separate_qkv_linear:
                linear_count += 1
                cur_id = self.update_for_separate_qkv_linear(cur_id, param_name, node_name)
            elif self.param_groups[cur_token_spelling] == NORM_PARAM and norm_count == 0:
                norm_count += 1
                cur_id = self.update_for_attention_norm(cur_id, param_name, node_name)
            elif self.param_groups[cur_token_spelling] == LINEAR_PARAM and linear_count == 0:
                linear_count += 1
                cur_id = self.update_for_qkv_linear(cur_id, param_name, node_name)
            elif self.is_output_linear(cur_token_spelling, node_name) and linear_count > 0:
                linear_count += 1
                cur_id = self.update_for_output_linear(cur_id, param_name, node_name)
            elif self.param_groups[cur_token_spelling] == MLP_PARAM:
                cur_id = self.update_for_mlp(cur_id, param_name, node_name)
            cur_id += 1
        return self.updates

    def get_atb_nodes_and_parameters(self):
        cur_id, atb_nodes, atb_node_params, atb_param_nodes = 0, [], {}, {}
        while cur_id < self.all_token_len:
            cur_token_spelling = self.all_tokens[cur_id].spelling
            if cur_token_spelling == "atb" and self.all_tokens[cur_id + 2].spelling == "Node":
                print_spelling(self.all_tokens[cur_id: cur_id + 5])
                if self.all_tokens[cur_id + 3].spelling == "&":
                    atb_node = self.all_tokens[cur_id + 4]
                    cur_id += 4
                else:
                    atb_node = self.all_tokens[cur_id + 3]
                    cur_id += 3
                atb_nodes.append(atb_node.spelling)
            if cur_token_spelling in atb_nodes and self.all_tokens[cur_id + 2].spelling == "operation":
                print_spelling(self.all_tokens[cur_id - 3: cur_id + 3])
                if self.all_tokens[cur_id - 1].spelling == "&":
                    atb_node_params[cur_token_spelling] = self.all_tokens[cur_id - 3].spelling
                    atb_param_nodes.setdefault(self.all_tokens[cur_id - 3].spelling, []).append(cur_token_spelling)
                else:
                    atb_node_params[cur_token_spelling] = self.all_tokens[cur_id - 2].spelling
                    atb_param_nodes.setdefault(self.all_tokens[cur_id - 2].spelling, []).append(cur_token_spelling)
                cur_id += 5
            else:
                cur_id += 1
        return atb_node_params, atb_param_nodes

    def get_atb_parameters_type(self):
        atb_parameters = set(self.atb_param_nodes.keys())
        cur_id, atb_param_types = 0, {}
        while cur_id < self.all_token_len:
            cur_token_spelling = self.all_tokens[cur_id].spelling
            if cur_token_spelling in atb_parameters and cur_token_spelling not in atb_param_types:
                print_spelling(self.all_tokens[cur_id - 3: cur_id + 3])
                atb_param_types[cur_token_spelling] = self.all_tokens[cur_id - 1].spelling
            cur_id += 1
        return atb_param_types

    def groupby_param_type(self):
        param_groups = {}
        for _, param_type in self.atb_param_types.items():
            param_type_lower = param_type.lower()
            if "mlp" in param_type_lower:
                param_groups[param_type] = MLP_PARAM
            elif "attention" in param_type_lower:
                param_groups[param_type] = ATTENTION_PARAM
            elif "norm" in param_type_lower:
                param_groups[param_type] = NORM_PARAM
            elif "linear" in param_type_lower:
                param_groups[param_type] = LINEAR_PARAM
            elif "parallelparam" in param_type_lower:
                param_groups[param_type] = LINEAR_PARAM
        return param_groups

    def get_all_norm_nodes(self):
        norm_nodes = []
        for param_name, node_names in self.atb_param_nodes.items():
            param_type = self.atb_param_types.get(param_name, None)
            param_group = self.param_groups.get(param_type, None)
            if param_group == NORM_PARAM:
                norm_nodes.extend(node_names)
        return norm_nodes

    def is_mlp_norm_node_using_attn_norm_param(self):
        if not self.norm_nodes:
            raise IndexError('norm_nodes is empty')
        attn_norm_param_name = self.atb_node_params.get(self.norm_nodes[0], 'Not exist')
        return any([
            self.atb_node_params.get(node, 'Not exist') == attn_norm_param_name for node in self.norm_nodes[1:]
            ])

    def seek_till_node_operation_line(self, cur_id, node_name):
        pre_end_line_token_id = cur_id
        while cur_id < self.total_id:
            cur_token_spelling = self.all_tokens[cur_id].spelling
            if cur_token_spelling == ";":
                pre_end_line_token_id = cur_id
            elif cur_token_spelling == node_name and self.all_tokens[cur_id + 2].spelling == "operation":
                pre_end_line_pos = self.all_tokens[pre_end_line_token_id].extent.end.offset
                pre_end_line_pos += self.contents[pre_end_line_pos:].find("\n")
                insert_start = insert_end = pre_end_line_pos + 1
                break
            cur_id += 1
        return insert_start, insert_end, cur_id + 2

    def seek_till_name(self, cur_id, name):
        while cur_id < self.total_id:
            if self.all_tokens[cur_id].spelling == name:
                break
            cur_id += 1
        return cur_id

    def seek_till_previous_semicolon(self, cur_id):
        while cur_id < self.total_id:
            if self.all_tokens[cur_id].spelling == ";":
                break
            cur_id -= 1
        return cur_id

    def insert_contents_for_attention_norm(self, param_name):
        scale_name = self.in_tensor_added_params[self.cur_param_index]
        offset_name = self.in_tensor_added_params[self.cur_param_index + 1]
        self.cur_param_index += 2

        insert_contents = f"{self.indent_prefix}{param_name}.normParam.quantInputScale = param.{scale_name};\n"
        insert_contents += f"{self.indent_prefix}{param_name}.normParam.quantInputOffset = param.{offset_name};\n"
        insert_contents += f"{self.indent_prefix}{param_name}.normParam.quantType = atb::infer::QUANT_INT8;\n"
        return insert_contents

    def insert_contents_for_mlp_norm(self, param_name, param_type):
        scale_name = self.in_tensor_added_params[self.cur_param_index]
        offset_name = self.in_tensor_added_params[self.cur_param_index + 1]
        self.cur_param_index += 2

        insert_contents = f"{self.indent_prefix}atb::infer::{param_type} {param_name};\n"
        insert_contents += f"{self.indent_prefix}{param_name}.normParam.quantInputScale = param.{scale_name};\n"
        insert_contents += f"{self.indent_prefix}{param_name}.normParam.quantInputOffset = param.{offset_name};\n"
        insert_contents += f"{self.indent_prefix}{param_name}.normParam.quantType = atb::infer::QUANT_INT8;\n"
        return insert_contents

    def insert_contents_for_mlp(self, param_name):
        scale_name = self.in_tensor_added_params[self.cur_param_index]
        offset_name = self.in_tensor_added_params[self.cur_param_index + 1]
        self.cur_param_index += 2

        elewise_type = "atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT"
        insert_contents = f"{self.indent_prefix}// add quant op\n"
        insert_contents += f"{self.indent_prefix}{param_name}.isBias = true;\n"
        insert_contents += f"{self.indent_prefix}{param_name}.isPack = false;\n"
        insert_contents += f"{self.indent_prefix}{param_name}.isQuant = true;\n"
        insert_contents += f"{self.indent_prefix}{param_name}.transposeB = true;\n"
        insert_contents += f"{self.indent_prefix}{param_name}.quantDownParam.quantType = atb::infer::QUANT_INT8;\n"
        insert_contents += f"{self.indent_prefix}{param_name}.quantDownParam.isQuantOp = true;\n"
        insert_contents += f"{self.indent_prefix}{param_name}.quantDownParam.elewiseType = {elewise_type};\n"
        insert_contents += f"{self.indent_prefix}{param_name}.quantDownParam.inputScale = param.{scale_name};\n"
        insert_contents += f"{self.indent_prefix}{param_name}.quantDownParam.inputOffset = param.{offset_name};\n"
        return insert_contents

    def insert_contents_for_qkv_linear(self, param_name):
        linear_type = "atb::infer::LinearType::LINEAR_INT8INT8_INT32_FP16"
        insert_contents = f"{self.indent_prefix}{param_name}.linearType = {linear_type};\n"
        insert_contents += f"{self.indent_prefix}{param_name}.hasBias = true;\n"
        return insert_contents

    def insert_contents_for_output_linear(self, param_name):
        scale_name = self.in_tensor_added_params[self.cur_param_index]
        offset_name = self.in_tensor_added_params[self.cur_param_index + 1]
        self.cur_param_index += 2

        elewise_type = "atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT"
        insert_contents = f"{self.indent_prefix}{param_name}.isBias = true;\n"
        insert_contents += f"{self.indent_prefix}{param_name}.isQuant = true;\n"
        insert_contents += f"{self.indent_prefix}{param_name}.transposeB = true;\n"
        insert_contents += f"{self.indent_prefix}{param_name}.quantParam.quantType = atb::infer::QUANT_INT8;\n"
        insert_contents += f"{self.indent_prefix}{param_name}.quantParam.isQuantOp = true; // add quant op\n"
        insert_contents += f"{self.indent_prefix}{param_name}.quantParam.elewiseType = {elewise_type};\n"
        insert_contents += f"{self.indent_prefix}{param_name}.quantParam.inputScale = param.{scale_name};\n"
        insert_contents += f"{self.indent_prefix}{param_name}.quantParam.inputOffset = param.{offset_name};\n"
        return insert_contents

    def update_intensor_id(self, cur_id, cur_in_tensor_added):
        insert_contents = ""
        insert_start = insert_end = self.all_tokens[cur_id].extent.start.offset
        while cur_id < self.total_id:
            cur_token_spelling = self.all_tokens[cur_id].spelling
            if cur_token_spelling == "}":
                insert_start = insert_end = self.all_tokens[cur_id].extent.start.offset
                insert_contents = ", " + cur_in_tensor_added
                break
            elif cur_token_spelling == IN_HOLDER:
                insert_start = self.all_tokens[cur_id].extent.start.offset
                insert_end = self.all_tokens[cur_id].extent.end.offset
                insert_contents = cur_in_tensor_added
                cur_id += 1
                break
            cur_id += 1

        print_update_info(insert_contents, insert_start, insert_end, cur_id)
        self.updates.append((insert_start, insert_end, insert_contents))
        return cur_id

    def update_for_attention_norm(self, cur_id, param_name, node_name):
        insert_contents = self.insert_contents_for_attention_norm(param_name)
        insert_start, insert_end, cur_id = self.seek_till_node_operation_line(cur_id, node_name)
        print_update_info(insert_contents, insert_start, insert_end, cur_id)
        self.updates.append((insert_start, insert_end, insert_contents))

        cur_id = self.seek_till_name(cur_id, node_name)
        cur_id = self.seek_till_name(cur_id, "inTensorIds")
        cur_id = self.update_intensor_id(cur_id, IN_BETA)
        return cur_id

    def update_for_mlp_norm(self, cur_id, param_name, node_name):
        mlp_param_name = "mlp" + param_name
        param_type = self.atb_param_types.get(param_name, 'Not exist')
        insert_contents = self.insert_contents_for_mlp_norm(mlp_param_name, param_type)
        cur_id = self.seek_till_previous_semicolon(cur_id)
        insert_start, insert_end, _ = self.seek_till_node_operation_line(cur_id, node_name)  # ignore this cur_id
        print_update_info(insert_contents, insert_start, insert_end, cur_id)
        self.updates.append((insert_start, insert_end, insert_contents))

        print_spelling(self.all_tokens[cur_id: cur_id + 3], info=f"param_name: {param_name}, current 3 tokens: ")
        cur_id = self.seek_till_name(cur_id, param_name)
        insert_start = self.all_tokens[cur_id].extent.start.offset
        insert_end = self.all_tokens[cur_id].extent.end.offset
        insert_contents = mlp_param_name
        print_update_info(insert_contents, insert_start, insert_end, cur_id)
        self.updates.append((insert_start, insert_end, insert_contents))

        cur_id = self.seek_till_name(cur_id, node_name)
        cur_id = self.seek_till_name(cur_id, "inTensorIds")
        cur_id = self.update_intensor_id(cur_id, IN_BETA)
        return cur_id

    def update_for_mlp(self, cur_id, param_name, node_name):
        insert_contents = self.insert_contents_for_mlp(param_name)
        insert_start, insert_end, cur_id = self.seek_till_node_operation_line(cur_id, node_name)
        print_update_info(insert_contents, insert_start, insert_end, cur_id)
        self.updates.append((insert_start, insert_end, insert_contents))

        cur_id = self.seek_till_name(cur_id, node_name)
        cur_id = self.seek_till_name(cur_id, "inTensorIds")
        while self.cur_intensor_enum_index < len(self.in_tensor_added_enums):  # Add all
            in_tensor_added = self.in_tensor_added_enums[self.cur_intensor_enum_index]
            self.cur_intensor_enum_index += 1
            cur_id = self.update_intensor_id(cur_id, in_tensor_added)
        return cur_id

    def update_for_sparse_linear_param(self, cur_id):
        insert_contents = "LinearSparseParam"
        insert_start = self.all_tokens[cur_id].extent.start.offset
        insert_end = self.all_tokens[cur_id].extent.end.offset
        self.updates.append((insert_start, insert_end, insert_contents))

        insert_contents = " = { false, true, 8, 8 }"
        insert_start = self.all_tokens[cur_id + 1].extent.end.offset
        cur_id = self.seek_till_name(cur_id, ";")
        insert_end = self.all_tokens[cur_id].extent.start.offset
        self.updates.append((insert_start, insert_end, insert_contents))

    def update_for_qkv_linear(self, cur_id, param_name, node_name):
        insert_contents = self.insert_contents_for_qkv_linear(param_name)
        insert_start, insert_end, cur_id = self.seek_till_node_operation_line(cur_id, node_name)
        print_update_info(insert_contents, insert_start, insert_end, cur_id)
        self.updates.append((insert_start, insert_end, insert_contents))

        cur_id = self.seek_till_name(cur_id, node_name)
        cur_id = self.seek_till_name(cur_id, "inTensorIds")
        in_tensor_added = self.in_tensor_added_enums[self.cur_intensor_enum_index]
        self.cur_intensor_enum_index += 1
        descale_insert_pos = len(self.updates)  # Record current updates position, needs to move INDEX here
        cur_id = self.update_intensor_id(cur_id, in_tensor_added)

        in_tensor_added = self.in_tensor_added_enums[self.cur_intensor_enum_index]
        if in_tensor_added.endswith(BIAS_SUFFIX):
            self.cur_intensor_enum_index += 1
            cur_id = self.update_intensor_id(cur_id, in_tensor_added)

        in_tensor_added = self.in_tensor_added_enums[self.cur_intensor_enum_index]
        if self.enable_sparse and in_tensor_added.endswith(INDEX_SUFFIX):
            self.cur_intensor_enum_index += 1
            cur_id = self.update_intensor_id(cur_id, in_tensor_added)
            # Move INDEX ahead -> [INDEX, DESCALE, BIAS]
            self.updates = self.updates[:descale_insert_pos] + self.updates[-1:] + self.updates[descale_insert_pos:-1]
        return cur_id

    def update_for_separate_qkv_linear(self, cur_id, param_name, node_name):
        cur_id = self.seek_till_name(cur_id, "inTensorIds")
        in_tensor_added = self.in_tensor_added_enums[self.cur_intensor_enum_index]
        self.cur_intensor_enum_index += 1
        descale_insert_pos = len(self.updates)  # Record current updates position, needs to move INDEX here
        cur_id = self.update_intensor_id(cur_id, in_tensor_added)

        in_tensor_added = self.in_tensor_added_enums[self.cur_intensor_enum_index]
        if in_tensor_added.endswith(BIAS_SUFFIX):
            self.cur_intensor_enum_index += 1
            cur_id = self.update_intensor_id(cur_id, in_tensor_added)

        in_tensor_added = self.in_tensor_added_enums[self.cur_intensor_enum_index]
        if self.enable_sparse and in_tensor_added.endswith(INDEX_SUFFIX):
            self.cur_intensor_enum_index += 1
            cur_id = self.update_intensor_id(cur_id, in_tensor_added)
            # Move INDEX ahead -> [INDEX, DESCALE, BIAS]
            self.updates = self.updates[:descale_insert_pos] + self.updates[-1:] + self.updates[descale_insert_pos:-1]
        return cur_id

    def update_for_output_linear(self, cur_id, param_name, node_name):
        insert_contents = self.insert_contents_for_output_linear(param_name)
        insert_start, insert_end, cur_id = self.seek_till_node_operation_line(cur_id, node_name)
        print_update_info(insert_contents, insert_start, insert_end, cur_id)
        self.updates.append((insert_start, insert_end, insert_contents))

        cur_id = self.seek_till_name(cur_id, node_name)
        cur_id = self.seek_till_name(cur_id, "inTensorIds")
        in_tensor_added = self.in_tensor_added_enums[self.cur_intensor_enum_index]
        self.cur_intensor_enum_index += 1
        cur_id = self.update_intensor_id(cur_id, in_tensor_added)

        in_tensor_added = self.in_tensor_added_enums[self.cur_intensor_enum_index]
        self.cur_intensor_enum_index += 1
        cur_id = self.update_intensor_id(cur_id, in_tensor_added)
        return cur_id

    def is_mlp_norm_node(self, cur_id):
        if self.all_tokens[cur_id].spelling not in self.norm_nodes:
            return False
        if self.all_tokens[cur_id + 2].spelling != "operation":
            return False
        return True

    def is_output_linear(self, cur_token_spelling, node_name):
        return self.param_groups.get(cur_token_spelling, 'Not exist') == LINEAR_PARAM and "out" in node_name.lower()

    def is_separate_qkv_node_operation(self, cur_token_spelling, cur_id):
        return cur_token_spelling in self.separate_qkv_nodes and self.all_tokens[cur_id + 2].spelling == "operation"
