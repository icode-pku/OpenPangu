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
import re
import sys

import json
from functools import reduce

from transformers import AutoConfig

from components.utils.file_utils import check_and_get_real_path, FileCheckConst
from msit_llm.transform.utils import write_file
from msit_llm.common.log import logger
from msit_llm.transform.torch_to_atb_python.utils import to_transformers_traced_module, get_valid_name, \
    get_lambda_source_code, generate_infer_file, get_config_attr, build_transformers_model, \
    find_mindie_supported_model, ATBModel, ATBModelConfig, Operation
from msit_llm.transform.torch_to_atb_python.env import NN_MODULE_STACK, SKIP_NODES, SKIP_MODULES, \
    TORCH_MODULE_TO_ATB_MAP, FX_OP_TYPES, FIXED_INPUTS, KV_CACHE_SURFFIX, BASIC_INPUT_NAMES, \
    RESHPAE_KIND, GATE_UP_WEIGHT, DOWN_WEIGHT


atb_speed_path = os.getenv("ATB_SPEED_HOME_PATH", None)
if not atb_speed_path:
    logger.warning("ATB_SPEED_HOME_PATH environment variable not valid, will skip build. Try install mindie")
else:
    sys.path.append(os.path.join(atb_speed_path, "lib"))


try:
    from _libatb_torch import _GraphOperation as GraphOperation  # May error
    from _libatb_torch import _BaseOperation as BaseOperation  # May error
except Exception:  # Could be import error or mindie errors, just catch all Exceptions
    logger.warning("import _libatb_torch failed, will skip build. Try install compatible mindie")
    GraphOperation, BaseOperation = None, None

# Force setting ASD print error log to stdout
if os.environ.get("ASDOPS_LOG_LEVEL") == "FATAL":
    os.environ["ASDOPS_LOG_LEVEL"] = "ERROR"
os.environ["ASDOPS_LOG_TO_STDOUT"] = "1"


class ATBModelFromTorch(ATBModel):
    """Create ATB model from raw transformers torch model
    >>> import torch, torch_npu
    >>> from transformers.models.llama import LlamaConfig, LlamaForCausalLM
    >>> from msit_llm.transform.torch_to_atb_python import ATBModel, ATBModelConfig, ATBModelFromTorch
    >>>
    >>> cc = LlamaConfig()
    >>> cc.num_hidden_layers, cc.hidden_size, cc.intermediate_size = 2, 1024, 4096  # smaller
    >>> mm = LlamaForCausalLM(cc).eval()
    >>> torch.save(mm.state_dict(), 'stat_dict.pt')  # Save for later usage
    >>> input_ids, position_ids = torch.arange(32), torch.arange(32)
    >>> with torch.no_grad():
    >>>     torch_out = mm(input_ids=input_ids[None], position_ids=position_ids[None]).logits
    >>>
    >>> atb_model = ATBModelFromTorch(mm)
    >>> atb_model.set_weights(mm.state_dict())
    >>> # Also set buffers in, which includes inv_freq. Ignore WARNINGs of missing weights
    >>> atb_model.set_weights(dict(mm.named_buffers()))
    >>> out = atb_model(input_ids=input_ids, position_ids=position_ids)
    >>> print({kk: vv.shape for kk, vv in out.items()})
    # 输出Size为{'output': torch.Size([32, 32000])}
    >>> print(torch.allclose(torch_out, out['output'].cpu().float(), atol=5e-2))
    # True
    >>> atb_model.to_file()  # Save atb model to a py file
    # 'llamaforcausallm_atb_float.py'

    Create ATB model from saved file and load saved weights
    >>> import torch, torch_npu, llamaforcausallm_atb_float
    >>> from msit_llm.transform.torch_to_atb_python import ATBModel
    >>> aa = ATBModel(llamaforcausallm_atb_float.Model(), {'vocab_size': 32000})
    >>> ss = torch.load('stat_dict.pt')  # Load from previously saved weights
    >>> aa.set_weights(ss)
    >>> # Set inv_freq for cos_table for sin_table. Ignore WARNINGs of missing weights
    >>> inv_freq = 1.0 / (1e4 ** (torch.arange(0, aa.atb_model.head_dim, 2) / aa.atb_model.head_dim))
    >>> aa.set_weights({'inv_freq': inv_freq})
    >>> out = atb_model(input_ids=input_ids, position_ids=position_ids)
    >>> print({kk: vv.shape for kk, vv in out.items()})
    # 输出Size为{'output': torch.Size([32, 32000])}
    """

    def __init__(
        self,
        torch_model,
        config=None,
        input_names=BASIC_INPUT_NAMES,
        max_batch_size=1,
        max_seq_len=1024,
        to_quant=False,
        quant_disable_names=None,
        is_vl_model=False,
        dtype="float16",
    ):
        self.torch_model, self.config, self.is_vl_model = torch_model, config, is_vl_model
        self.to_quant, self.quant_disable_names = to_quant, quant_disable_names or ()
        self.config = getattr(torch_model, "config", None) if config is None else config
        if self.config is None:
            raise ValueError("Either config or torch_model.config shold be not empty one")

        self.num_attention_heads = get_config_attr(self.config, "num_attention_heads")
        if self.num_attention_heads is None or self.num_attention_heads <= 0:
            raise ValueError("Invalid config, num_attention_heads not exists or < 0")

        self.hidden_size = get_config_attr(self.config, "hidden_size")
        if self.hidden_size is None or self.hidden_size <= 0:
            raise ValueError("Invalid config, hidden_size not exists or < 0")

        self.vocab_size = get_config_attr(self.config, "vocab_size", -1)
        if self.vocab_size is None or self.vocab_size <= 0:
            logger.warning("vocab_size from config not exists or < 0. Ignore if this is intended")

        self.head_dim = self.hidden_size // self.num_attention_heads
        self.traced_module = to_transformers_traced_module(torch_model, input_names=input_names)
        self.model_name = get_valid_name(torch_model.__class__.__name__)

        self.weight_names, self.weight_stack_map = list(self.traced_module.state_dict().keys()), {}
        for ii in self.weight_names:
            self.weight_stack_map.setdefault(".".join(ii.split(".")[:-1]), []).append(ii)

        self.num_key_value_heads = get_config_attr(self.config, "num_key_value_heads", default=self.num_attention_heads)
        if not isinstance(self.num_key_value_heads, int):
            raise ValueError("num_key_value_heads must be a integer")
        self.rope_theta = get_config_attr(self.config, "rope_theta", default=1e4)
        if not isinstance(self.rope_theta, (float, int)):
            raise ValueError("rope_theta must be a floating point number")
        self.rms_norm_eps = get_config_attr(self.config, "rms_norm_eps", default=1e-5)
        if not isinstance(self.rms_norm_eps, (float, int)):
            raise ValueError("rms_norm_eps must be a floating point number")
        self.topk = get_config_attr(self.config, "num_experts_per_tok", default=0)
        if not isinstance(self.topk, int):
            raise ValueError("topk must be a integer")
        self.num_experts = get_config_attr(self.config, "num_local_experts", default=8)
        if not isinstance(self.num_experts, int):
            raise ValueError("num_experts must be a integer")
        self.model_type = get_config_attr(self.config, "torch_dtype", default="bfloat16")
        if not isinstance(self.model_type, int):
            raise ValueError("model_type must be a integer")
        self.num_layers = get_config_attr(self.config, "num_hidden_layers", default=8)
        if not isinstance(self.num_layers, int):
            raise ValueError("num_layers must be a integer")

        self.torch_module_to_atb_map = {}
        for kk, vv in TORCH_MODULE_TO_ATB_MAP.items():
            re_key = re.compile(kk)
            if vv.get("op_type", None) == "SelfAttention" and "op_param" in vv:
                vv["op_param"].update({"headNum": self.num_attention_heads, "kvHeadNum": self.num_key_value_heads})
                vv["op_param"].update({"qkScale": 1 / (float(self.head_dim) ** 0.5)})
            elif vv.get("op_type", None) == "RmsNorm" and "op_param" in vv:
                vv["op_param"].update({"epsilon": self.rms_norm_eps})
            self.torch_module_to_atb_map[re_key] = Operation(**vv)

        self.pre_qkv_name, self.pre_query_name, self.pre_key_name, self.pre_value_name = "", "", "", ""
        self.is_apply_rope = False
        self.model_inputs, self.model_outputs, self.operations = [], [], []
        # base graph is set execute_as_single=False, has to keep all operaions as property
        self.base_graph_operations, self.k_cache_names, self.v_cache_names = [], [], []

        self.is_moe = self.find_moe()
        self.convert_fx_traced_module()
        if to_quant:
            logger.info(f"calling convert_to_quant, quant_disable_names = {self.quant_disable_names}")
            self.convert_to_quant()
        self.to_file()

        self.atb_model_config = ATBModelConfig(
            vocab_size=self.vocab_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        if GraphOperation is None or BaseOperation is None:
            logger.warning("Will skip build")
        else:
            self.atb_model = self.build_atb_model()
            super().__init__(atb_model=self.atb_model, atb_model_config=self.atb_model_config, dtype=dtype)

    @staticmethod
    def get_cur_repeat_block_idx(module_name):
        for tok in module_name.split("."):
            if str.isdigit(tok):
                return int(tok)
        return -1

    @staticmethod
    def _get_module_type_by_nn_module_stack(node):
        node_module_stack = list(node.meta[NN_MODULE_STACK].values())
        return None if len(node_module_stack) == 0 else node_module_stack[-1].__name__

    @staticmethod
    def _get_module_name_by_nn_module_stack(node):
        node_module_stack = list(node.meta[NN_MODULE_STACK].keys())
        return None if len(node_module_stack) == 0 else node_module_stack[-1]

    @staticmethod
    def _should_skip_node(node):
        if node.op == FX_OP_TYPES.call_method and node.target in SKIP_NODES:
            return True
        if node.op == FX_OP_TYPES.call_function and node.target.__name__ in SKIP_NODES:
            return True
        if node.op == FX_OP_TYPES.call_function and node.meta.get("is_wrapped", False):
            return True
        return False

    def to_file(self, output_file=None):
        indent = " " * 4
        stacked_operations, stacked_inputs, stacked_outputs = self._stack_operations()
        base_model_name = self.model_name

        contents = [
            "import os",
            "import sys",
            "import json",
            "import torch",
            "import torch_npu",
            "from functools import reduce",
            "",
            "atb_speed_path = os.getenv('ATB_SPEED_HOME_PATH')",
            "if not atb_speed_path:",
            f"{indent}raise OSError('ATB_SPEED_HOME_PATH environment variable not valid. Try install mindie')",
            "sys.path.append(os.path.join(atb_speed_path, 'lib'))",
            "from _libatb_torch import _GraphOperation as GraphOperation",
            "from _libatb_torch import _BaseOperation as BaseOperation",
            "",
            "if os.environ.get('ASDOPS_LOG_LEVEL') == 'FATAL':",
            f"{indent}os.environ['ASDOPS_LOG_LEVEL'] = 'ERROR'",
            "os.environ['ASDOPS_LOG_TO_STDOUT'] = '1'  # Force setting ASD printing error log to stdout",
            "",
            "class Model(GraphOperation):",
            f"{indent}def __init__(self, outputs=None):",
            f"{indent * 2}self.model_name = '{base_model_name}'",
            f"{indent * 2}super().__init__(self.model_name)",
            "",
            f"{indent * 2}self.num_attention_heads, self.head_dim = {self.num_attention_heads}, {self.head_dim}",
            f"{indent * 2}self.num_key_value_heads, self.vocab_size = {self.num_key_value_heads}, {self.vocab_size}",
            f"{indent * 2}self.rope_theta = {self.rope_theta}",
        ]
        if "Mixtral" in base_model_name:
            contents.append(f"{indent * 2}self.topk = {self.topk}")
            contents.append(f"{indent * 2}self.num_layers = {self.num_layers}")
            contents.append(f"{indent * 2}self.num_experts = {self.num_experts}")

        def _get_input_output_name(graph_name):
            return f"{graph_name}_inputs", f"{graph_name}_outputs"

        def _to_file(graph_name, operations, depth=0):
            contents.append("")
            if depth > 0:
                contents.append(f"{indent * 2}{graph_name} = GraphOperation('{graph_name}')")
                this_name = graph_name
            else:
                this_name = "self"

            graph_inputs, graph_outputs = stacked_inputs.pop(0), stacked_outputs.pop(0)
            contents.append(f"{indent * 2}{graph_name}_inputs = [")
            for ii in sorted(graph_inputs, key=lambda xx: "0" if xx in FIXED_INPUTS else xx):
                contents.append(f"{indent * 3}'{ii}',")
            contents.append(f"{indent * 2}]")

            if depth > 0:
                contents.append(f"{indent * 2}{graph_name}_outputs = {graph_outputs}")
            else:
                contents.append(f"{indent * 2}{graph_name}_outputs = outputs or {graph_outputs}")
            cur_inputs, cur_outputs = _get_input_output_name(graph_name)
            contents.append(f"{indent * 2}{this_name}.add_input_output(input={cur_inputs}, output={cur_outputs})")
            contents.append("")

            sub_graph_id = 0
            for op in operations:
                if isinstance(op, list):
                    sub_graph_name = f"sub_graph_{sub_graph_id}"
                    _to_file(sub_graph_name, op, depth=depth + 1)
                    cur_inputs, cur_outputs = _get_input_output_name(sub_graph_name)
                    contents.append(
                        f"{indent * 2}{this_name}.add_operation({sub_graph_name}, {cur_inputs}, {cur_outputs})"
                    )
                    contents.append(f"{indent * 2}{this_name}.{sub_graph_name} = {sub_graph_name}")
                    contents.append("")
                    sub_graph_id += 1
                elif op.op_type == "add_reshape":
                    function = get_lambda_source_code(op.function)
                    contents.append(
                        f"{indent * 2}{this_name}.add_reshape('{op.inputs[0]}', '{op.outputs[0]}', {function})"
                    )
                else:
                    op_kwargs = f"op_type='{op.op_type}', op_param='{json.dumps(op.op_param)}', op_name='{op.op_name}'"

                    if depth == 0:
                        cur_op = f"{this_name}." + op.op_name.replace(".", "_")
                        contents.append(f"{indent * 2}{cur_op} = BaseOperation({op_kwargs})")
                    else:
                        cur_op = f"BaseOperation({op_kwargs})"
                    contents.append(f"{indent * 2}{this_name}.add_operation(")
                    contents.append(f"{indent * 3}operation={cur_op},")
                    contents.append(f"{indent * 3}input={op.inputs},")
                    contents.append(f"{indent * 3}output={op.outputs},")
                    contents.append(f"{indent * 2})")

            contents.append("")
            contents.append(f"{indent * 2}{this_name}.execute_as_single = {False if depth == 0 else True}")
            contents.append(f"{indent * 2}{this_name}.build()")

        _to_file(base_model_name, stacked_operations)
        contents.append("")
        contents_str = "\n".join(contents)

        if output_file is None:
            output_file = "_".join([base_model_name.lower(), "atb", "quant.py" if self.to_quant else "float.py"])
        elif not output_file.endswith(".py"):
            output_file += ".py"
        write_file(output_file, contents_str)
        return output_file

    def find_moe(self):
        for key in list(self.traced_module.state_dict().keys()):
            if "moe.experts" in key:
                return True
        return False

    def convert_fx_traced_module(self):
        previous_module_name, cur_module_name, previous_operation_out, base_module_name = None, None, None, None
        input_node_map, output_node_map, operation_outputs = {}, {}, {}

        if self.is_moe:
            for i in range(self.num_layers):
                self.model_inputs += [GATE_UP_WEIGHT + str(i), DOWN_WEIGHT + str(i)]
            self._op_process_rope()

        for node in self.traced_module.graph.nodes:
            logger.debug("=" * 30 + "\n")
            logger.debug(f"node.name = {node.name}, node.op = {node.op}, node.meta = {node.meta}")
            logger.debug(f"node.all_input_nodes = {node.all_input_nodes}, node.target = {node.target}")
            if node.op == FX_OP_TYPES.placeholder:  # Input node
                self.model_inputs.append(node.name)
                input_node_map[node.name] = [node.name]
                output_node_map[node.name] = [node.name]
                continue
            if node.op == FX_OP_TYPES.output:  # Output node
                self.model_outputs.append(node.name)
                continue
            if not hasattr(node, "meta") or not node.meta.get(NN_MODULE_STACK, []):
                if node.op == FX_OP_TYPES.call_function:
                    output_node_map[node.name] = previous_operation_out  # op like getitem, set to previous output
                continue

            if base_module_name is None:
                base_module_name = list(node.meta[NN_MODULE_STACK].keys())[0]

            cur_module_name = self._get_module_name_by_nn_module_stack(node)
            # do rope in a unified manner
            if (cur_module_name.endswith("self_attn.rotary_emb") and
                    previous_module_name.endswith("self_attn") and self.is_moe):
                continue
            input_node_map.setdefault(cur_module_name, []).extend([ii.name for ii in node.all_input_nodes])

            logger.debug(f"cur_module_name = {cur_module_name}, previous_module_name = {previous_module_name}")
            if cur_module_name != base_module_name:
                output_node_map[node.name] = previous_operation_out  # will be overwriten later
            if cur_module_name == previous_module_name:
                continue
            if self._should_skip_node(node):
                logger.debug(f"Current node skipped: {node.name}")
                continue
            if node.op == FX_OP_TYPES.call_module and self._get_module_type_by_nn_module_stack(node) in SKIP_MODULES:
                logger.debug(f"Current module skipped: {node.name}")
                cur_module_name = previous_module_name  # Module like Dropout skipped, set back to previous name
                continue
            previous_module_name = cur_module_name

            logger.debug(f"cur_module_name = {cur_module_name}, node.name = {node.name}")
            node_module_type, cur_inputs, module_name = self._get_node_type_and_inputs_and_name(node, output_node_map)
            logger.debug(f"node_module_type={node_module_type}, cur_inputs={cur_inputs}, module_name={module_name}")
            if self._find_in_torch_module_to_atb_map(node_module_type) is None:
                logger.warning(f"node not supported: node.name = {node.name}, node.type = {node_module_type}")
                continue
            if len(cur_inputs) == 0:
                logger.warning(f"found none valid input: node.name = {node.name}, node.type = {node_module_type}")
                continue

            atb_operation = self._convert_module(node_module_type, module_name, cur_inputs)
            self._check_and_set_pre_qkv_name(atb_operation)
            logger.debug(f"atb_operation = {atb_operation.to_json()}")
            if not self.is_apply_rope and atb_operation.op_type == "Rope":
                self._op_process_rope(atb_operation=atb_operation, module_name=module_name)
            elif atb_operation.op_type == "MixtralSparseMoeBlock":
                self._op_process_moe_gate(atb_operation=atb_operation, module_name=module_name)
                self._op_process_moe_router(atb_operation=atb_operation, module_name=module_name)
                self._op_process_moe_norm(atb_operation=atb_operation, module_name=module_name)
                find_names = re.findall(r'\d+', module_name)
                if len(find_names) < 1:
                    raise RuntimeError(f"Invalid module name: {module_name}")
                layer_id = find_names[0]
                self._op_process_moe_mlp_init_routing(atb_operation=atb_operation, module_name=module_name)
                self._op_process_moe_mlp_cast(atb_operation=atb_operation, module_name=module_name)
                self._op_process_moe_mlp_gate_up_gmm(atb_operation=atb_operation,
                                                     module_name=module_name,
                                                     layer_id=layer_id)
                self._op_process_moe_mlp_activation_block(atb_operation=atb_operation, module_name=module_name)
                self._op_process_moe_mlp_down_gmm(atb_operation=atb_operation,
                                                  module_name=module_name,
                                                  layer_id=layer_id)
                self._op_process_moe_mlp_moe_token_unpermute(atb_operation=atb_operation, module_name=module_name)
                if len(self.operations) < 1:
                    raise RuntimeError("Build operations failed, Please check it!")
                cur_outputs = self.operations[-1].outputs
                output_node_map[node.name] = previous_operation_out = operation_outputs[cur_module_name] = cur_outputs
            elif atb_operation.op_type == "SelfAttention":
                self._op_process_attention(atb_operation=atb_operation, module_name=module_name)
                cur_outputs = self.operations[-1].outputs
                output_node_map[node.name] = previous_operation_out = operation_outputs[cur_module_name] = cur_outputs
            elif atb_operation.op_type in ["Linear", "LinearParallel"]:
                self._op_process_linear(atb_operation=atb_operation, module_name=module_name)
                cur_outputs = self.operations[-1].outputs
                output_node_map[node.name] = previous_operation_out = operation_outputs[cur_module_name] = cur_outputs
            else:
                self.operations.append(atb_operation)
                cur_outputs = atb_operation.outputs
                output_node_map[node.name] = previous_operation_out = operation_outputs[cur_module_name] = cur_outputs

        self._refine_inputs_outputs(input_node_map, output_node_map, operation_outputs)
        if self.is_vl_model:
            self._replace_input_ids_by_inputs_embeds_for_vl_model()
        self.operations[-1].outputs = self.model_outputs
        return self.model_inputs, self.model_outputs, self.weight_names, self.operations

    def convert_to_quant(self):
        # Has to split out from convert_fx_traced_module, needs actual Linear input names
        quant_disable_names = set([ii for ii in self.quant_disable_names if ii is not None and len(ii) > 0])
        operations_with_quant = []
        for _, op in enumerate(self.operations):
            logger.debug(f"op.op_name = {op.op_name}")
            if op.op_type not in ["Linear", "LinearParallel"]:
                operations_with_quant.append(op)
                continue
            if op.op_name in quant_disable_names:
                operations_with_quant.append(op)
                quant_disable_names.remove(op.op_name)
                continue

            logger.debug(f"Convert {op.op_name} to quant node")
            linear_quant_weights = []
            bias, deq_scale = f"{op.op_name}.bias", f"{op.op_name}.deq_scale"
            input_scale, input_offset = f"{op.op_name}.input_scale", f"{op.op_name}.input_offset"
            self.weight_names += [input_scale, input_offset, deq_scale]

            if bias not in op.inputs:
                linear_quant_weights.append(bias)
                self.weight_names.append(bias)
            linear_quant_weights.append(deq_scale)

            elewise_quant_node = Operation(
                op_type="Elewise",
                op_param={"elewiseType": "ELEWISE_QUANT_PER_CHANNEL"},
                op_name=f"{op.op_name}.elewise_quant",
                inputs=[f"{op.inputs[0]}", input_scale, input_offset],
                outputs=[f"{op.op_name}.elewise_quant.out"],
            )

            op.op_param.update({"transposeA": False, "transposeB": True, "hasBias": True, "outDataType": "ACL_FLOAT16"})
            op.inputs = elewise_quant_node.outputs + op.inputs[1:] + linear_quant_weights

            operations_with_quant += [elewise_quant_node, op]
        self.operations = operations_with_quant

        if len(quant_disable_names) > 0:
            logger.warning(f"Some layers in quant_disable_names not found in model: {quant_disable_names}")

    def build_atb_model(self):
        stacked_operations, stacked_inputs, stacked_outputs = self._stack_operations()

        def _build_atb_model(graph_name, operations, depth=0):
            atb_model = GraphOperation(graph_name)
            graph_inputs, graph_outputs = stacked_inputs.pop(0), stacked_outputs.pop(0)
            logger.debug(f"graph_name = {graph_name}")
            logger.debug(f"len(graph_inputs) = {len(graph_inputs)}, graph_inputs = {graph_inputs}")
            logger.debug(f"len(graph_outputs) = {len(graph_outputs)}, graph_outputs = {graph_outputs}")
            logger.debug(f"operations: {[ii.op_name for ii in operations if not isinstance(ii, list)]}")

            atb_model.add_input_output(input=graph_inputs, output=graph_outputs)
            sub_graph_id = 0
            for op in operations:
                if isinstance(op, list):
                    sub_graph_name = f"sub_graph_{sub_graph_id}"
                    cur_operation = _build_atb_model(sub_graph_name, op, depth=depth + 1)
                    atb_model.add_operation(cur_operation, cur_operation.input_names, cur_operation.output_names)
                    sub_graph_id += 1
                elif op.op_type == "add_reshape":
                    atb_model.add_reshape(op.inputs[0], op.outputs[0], op.function)
                    continue
                else:
                    cur_operation = BaseOperation(
                        op_type=op.op_type, op_param=json.dumps(op.op_param), op_name=op.op_name
                    )
                    atb_model.add_operation(operation=cur_operation, input=op.inputs, output=op.outputs)
                if depth == 0:
                    self.base_graph_operations.append(cur_operation)
            atb_model.execute_as_single = False if depth == 0 else True
            atb_model.build()
            return atb_model

        atb_model = _build_atb_model(self.model_name, stacked_operations)
        return atb_model

    def _get_node_type_and_inputs_and_name(self, node, output_node_map=None):
        output_node_map = output_node_map or {}
        cur_module_name = self._get_module_name_by_nn_module_stack(node)
        cur_model_type = self._get_module_type_by_nn_module_stack(node)
        if node.op == FX_OP_TYPES.call_function and node.target.__name__ == "mul" and "gelu" in cur_model_type.lower():
            cur_inputs = [None]
            module_name = cur_module_name
        elif node.op == FX_OP_TYPES.call_function and self._find_in_torch_module_to_atb_map(node.target.__name__):
            cur_model_type = node.target.__name__
            # No other inputs if function
            cur_inputs = []
            for ii in node.all_input_nodes:
                if ii.name not in output_node_map:
                    continue
                cur_inputs += output_node_map[ii.name]
            module_name = cur_module_name + "." + node.name
        else:
            # None marks for placeholder of other inputs
            cur_inputs = self.weight_stack_map.get(cur_module_name, []) + [None]
            module_name = cur_module_name
        return cur_model_type, cur_inputs, module_name

    def _check_and_set_pre_qkv_name(self, atb_operation):
        if atb_operation.op_type == "Linear":
            sub_name = atb_operation.op_name.split(".")[-1]
            logger.debug(f"Checking if qkv Linear: {atb_operation}")
            if all(sub in sub_name for sub in ("q", "k", "v")):
                self.pre_qkv_name = atb_operation.outputs[0]
            elif "q" in sub_name:
                self.pre_query_name = atb_operation.outputs[0]
            elif "k" in sub_name:
                self.pre_key_name = atb_operation.outputs[0]
            elif "v" in sub_name:
                self.pre_value_name = atb_operation.outputs[0]

    def _find_in_torch_module_to_atb_map(self, node_type):
        for kk, vv in self.torch_module_to_atb_map.items():
            if kk.fullmatch(node_type):
                return vv.copy()
        return None

    def _convert_module(self, node_module_type, node_module_name, input_names):
        atb_operation = self._find_in_torch_module_to_atb_map(node_module_type)
        outputs = [node_module_name + ".out"]
        atb_operation.op_name = node_module_name
        atb_operation.inputs = getattr(atb_operation, "inputs", []) + input_names
        atb_operation.outputs = getattr(atb_operation, "outputs", []) + outputs
        return atb_operation

    def _op_process_linear(self, atb_operation=None, module_name=""):
        bias_name = f"{atb_operation.op_name}.bias"
        if bias_name in self.weight_names:
            atb_operation.op_param.update({"hasBias": True})
        self.operations.append(atb_operation)

    def _op_process_rope(self, atb_operation=None, module_name=""):
        self.is_apply_rope = True
        self.model_inputs += [FIXED_INPUTS.cos_table, FIXED_INPUTS.sin_table]
        self.operations.append(
            self._convert_module("Gather", "gather_cos", [FIXED_INPUTS.cos_table, FIXED_INPUTS.position_ids])
        )
        self.operations.append(
            self._convert_module("Gather", "gather_sin", [FIXED_INPUTS.sin_table, FIXED_INPUTS.position_ids])
        )

        if FIXED_INPUTS.position_ids not in self.model_inputs:
            self.model_inputs += [FIXED_INPUTS.position_ids]

    def _op_process_moe_gate(self, atb_operation=None, module_name=""):
        self.operations += [
            Operation(
                op_type="Linear",
                op_param={"hasBias": False, "enAccum": False},
                inputs=[module_name.split("block_sparse_moe")[0] + "post_attention_layernorm.out",
                        module_name + ".gate.weight"],
                outputs=[module_name + ".intermediate_router_logits"],
                op_name=module_name + ".gate",
            ),
        ]

    def _op_process_moe_router(self, atb_operation=None, module_name=""):
        self.operations += [
            Operation(
                op_type="Softmax",
                op_param={"axes": [1]},
                inputs=[module_name + ".intermediate_router_logits"],
                outputs=[module_name + ".intermediate_router_weights"],
                op_name=module_name + ".router_softmax",
            ),
            Operation(
                op_type="Sort",
                op_param={"num": [self.topk]},
                inputs=[module_name + ".intermediate_router_weights"],
                outputs=[module_name + ".intermediate_router_weights_topk",
                         module_name + ".intermediate_selected_experts"],
                op_name=module_name + ".router_topk",
            ),
        ]

    def _op_process_moe_norm(self, atb_operation=None, module_name=""):
        self.operations += [
            Operation(
                op_type="Reduce",
                op_param={"reduceType": "REDUCE_SUM", "axis": [1]},
                inputs=[module_name + ".intermediate_router_weights_topk"],
                outputs=[module_name + ".intermediate_router_weights_topk_sumed0"],
                op_name=module_name + ".norm_sum",
            ),
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], 1],
                inputs=[module_name + ".intermediate_router_weights_topk_sumed0"],
                outputs=[module_name + ".intermediate_router_weights_topk_sumed1"],
                op_name=module_name + ".norm_reshape",
            ),
            Operation(
                op_type="Elewise",
                op_param={"elewiseType": "ELEWISE_REALDIV"},
                inputs=[module_name + ".intermediate_router_weights_topk",
                        module_name + ".intermediate_router_weights_topk_sumed1"],
                outputs=[module_name + ".intermediate_router_weights_topk_reduced"],
                op_name=module_name + ".norm_div",
            ),
        ]

    def _op_process_moe_mlp_init_routing(self, atb_operation=None, module_name=""):
        self.operations += [
            Operation(
                op_type="MoeInitRouting",
                op_param={"topkNum": self.topk, "expertNum": self.num_experts},
                inputs=[module_name.split("block_sparse_moe")[0] + "post_attention_layernorm.out",
                        module_name + ".intermediate_selected_experts"],
                outputs=[module_name + ".intermediate_sorted_hidden_states",
                         module_name + ".intermediate_idx",
                         module_name + ".intermediate_group_list"],
                op_name=module_name + ".moe_init_routing",
            )
        ]

    def _op_process_moe_mlp_cast(self, atb_operation=None, module_name=""):
        self.operations += [
            Operation(
                op_type="Elewise",
                op_param={"elewiseType": "ELEWISE_CAST", 'outTensorType': 'ACL_INT64'},
                inputs=[module_name + ".intermediate_group_list"],
                outputs=[module_name + ".intermediate_group_list_int64"],
                op_name=module_name + ".elewise_cast",
            )
        ]

    def _op_process_moe_mlp_gate_up_gmm(self, atb_operation=None, module_name="", layer_id=0):
        self.operations += [
            Operation(
                op_type="GroupedMatmul",
                op_param={"transposeB": False,
                          'outTensorType': 'ACL_BF16' if self.model_type == "bfloat16" else 'ACL_FLOAT16'},
                inputs=[module_name + ".intermediate_sorted_hidden_states",
                        GATE_UP_WEIGHT+layer_id,
                        module_name + ".intermediate_group_list_int64"],
                outputs=[module_name + ".intermediate_matmul_gate_up_out"],
                op_name=module_name + ".integrated_gmm_gate_up",
            )
        ]

    def _op_process_moe_mlp_activation_block(self, atb_operation=None, module_name=""):
        self.operations += [
            Operation(
                op_type="Split",
                op_param={"splitDim": 1, "splitNum": 2},
                op_name=module_name + ".activation_split",
                inputs=[module_name + ".intermediate_matmul_gate_up_out"],
                outputs=[module_name + ".intermediate_matmul_gate_out",
                         module_name + ".intermediate_matmul_up_out"]
            ),
            Operation(
                op_type="Activation",
                op_param={'activationType': 'ACTIVATION_SWISH'},
                op_name=module_name + ".activation",
                inputs=[module_name + ".intermediate_matmul_gate_out"],
                outputs=[module_name + ".intermediate_swish_out_internal"]
            ),
            Operation(
                op_type="Elewise",
                op_param={'elewiseType': 'ELEWISE_MUL'},
                op_name=module_name + ".activation_elewise_mul",
                inputs=[module_name + ".intermediate_swish_out_internal",
                        module_name + ".intermediate_matmul_up_out"],
                outputs=[module_name + ".intermediate_swish_out"]
            )
        ]

    def _op_process_moe_mlp_down_gmm(self, atb_operation=None, module_name="", layer_id=0):
        self.operations += [
            Operation(
                op_type="GroupedMatmul",
                op_param={"transposeB": False,
                          'outTensorType': 'ACL_BF16' if self.model_type == "bfloat16" else 'ACL_FLOAT16'},
                inputs=[module_name + ".intermediate_swish_out",
                        DOWN_WEIGHT+layer_id,
                        module_name + ".intermediate_group_list_int64"],
                outputs=[module_name + ".intermediate_mlp_out"],
                op_name=module_name + ".integrated_gmm_down",
            )
        ]

    def _op_process_moe_mlp_moe_token_unpermute(self, atb_operation=None, module_name=""):
        self.operations += [
            Operation(
                op_type="MoeTokenUnpermute",
                op_param={},
                op_name=module_name + ".moe_token_unpermute",
                inputs=[module_name + ".intermediate_mlp_out",
                        module_name + ".intermediate_idx",
                        module_name + ".intermediate_router_weights_topk_reduced"],
                outputs=[module_name + ".mlp_out"]
            )
        ]

    def _op_process_attention(self, atb_operation=None, module_name=""):
        atb_operation.inputs = [
            module_name + ".q_embed_",
            module_name + ".k_embed_",
            module_name + ".v_embed_",
            FIXED_INPUTS.attention_mask,
            FIXED_INPUTS.seq_len,
        ]

        query_name, key_name, value_name = self.pre_query_name, self.pre_key_name, self.pre_value_name
        if self.pre_qkv_name:
            logger.debug(f"Got stacked QKV Linear: {self.pre_qkv_name}")
            inputs = self.pre_qkv_name
            query_name, key_name, value_name = module_name + ".q", module_name + ".k", module_name + ".v"
            self.operations.append(
                Operation(
                    op_type="Split",
                    op_param={"splitDim": 1, "splitNum": 3},
                    inputs=[inputs],
                    outputs=[query_name, key_name, value_name],
                    op_name=module_name + ".split",
                )
            )
        # Set back to default value
        self.pre_qkv_name, self.pre_query_name, self.pre_key_name, self.pre_value_name = "", "", "", ""

        if self.is_apply_rope:
            inputs = [query_name, key_name, "gather_cos.out", "gather_sin.out", FIXED_INPUTS.seq_len]
            query_name, key_name = module_name + ".q_embed", module_name + ".k_embed"
            self.operations.append(
                Operation(
                    op_type="Rope",
                    op_param={"rotaryCoeff": 2},
                    inputs=inputs,
                    outputs=[query_name, key_name],
                    op_name=module_name + ".rope",
                )
            )

        k_cache_name = module_name + "." + KV_CACHE_SURFFIX.k_cache
        v_cache_name = module_name + "." + KV_CACHE_SURFFIX.v_cache
        reshape_and_cache_inputs = [
            module_name + ".k_embed_",
            module_name + ".v_embed_",
            k_cache_name,
            v_cache_name,
            FIXED_INPUTS.slots_mapping,
        ]
        self.k_cache_names.append(k_cache_name)
        self.v_cache_names.append(v_cache_name)
        self.operations += [
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], self.num_attention_heads, self.head_dim],
                inputs=[query_name],
                outputs=[module_name + ".q_embed_"],
                op_name=module_name + ".q." + RESHPAE_KIND.reshape_qkv,
            ),
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], self.num_key_value_heads, self.head_dim],
                inputs=[key_name],
                outputs=[module_name + ".k_embed_"],
                op_name=module_name + ".k." + RESHPAE_KIND.reshape_qkv,
            ),
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], self.num_key_value_heads, self.head_dim],
                inputs=[value_name],
                outputs=[module_name + ".v_embed_"],
                op_name=module_name + ".v." + RESHPAE_KIND.reshape_qkv,
            ),
            Operation(
                op_type="ReshapeAndCache",
                op_param={},
                inputs=reshape_and_cache_inputs,
                outputs=[k_cache_name, v_cache_name],
                op_name=module_name + ".reshape_and_cache",
            ),
            atb_operation,
            Operation(
                op_type="add_reshape",
                function=lambda org_shape: [org_shape[0], reduce(lambda xx, yy: xx * yy, org_shape[1:])],
                inputs=atb_operation.outputs,
                outputs=[ii + "_" for ii in atb_operation.outputs],
                op_name=module_name + "." + RESHPAE_KIND.reshape_0_12,
            ),
        ]

        if FIXED_INPUTS.slots_mapping not in self.model_inputs:
            self.model_inputs += [
                FIXED_INPUTS.slots_mapping,
                FIXED_INPUTS.attention_mask,
                FIXED_INPUTS.seq_len,
            ]
        self.model_inputs += [k_cache_name, v_cache_name]

    def _refine_inputs_outputs(self, input_node_map, output_node_map, operation_outputs):
        gathered_module_inputs = {}
        for kk, vv in input_node_map.items():
            gathered_inputs = []
            for ii in vv:
                if ii in output_node_map:
                    gathered_inputs += output_node_map[ii]
            for ii in set(gathered_inputs):
                if ii == kk or ii == kk + ".out":
                    continue
                gathered_module_inputs.setdefault(kk, []).extend(
                    operation_outputs.get(ii, ii if isinstance(ii, list) else [ii])
                )

        for op in self.operations:
            if not isinstance(op, list) and (len(op.inputs) == 0 or op.inputs[-1] is not None):
                continue

            op.inputs = op.inputs[:-1]  # Exclude last None
            if op.is_weights_first:
                op.inputs = op.inputs + gathered_module_inputs.get(op.op_name, [])
            else:
                op.inputs = gathered_module_inputs.get(op.op_name, []) + op.inputs

    def _replace_input_ids_by_inputs_embeds_for_vl_model(self):
        if FIXED_INPUTS.input_ids in self.model_inputs:
            self.model_inputs = [
                FIXED_INPUTS.inputs_embeds
                if ii == FIXED_INPUTS.input_ids else ii
                for ii in self.model_inputs
            ]

        embed_op_id, embed_outputs = -1, None
        for op_id, op in enumerate(self.operations):
            if len(op.inputs) == 2 and len(op.outputs) == 1 and FIXED_INPUTS.input_ids in op.inputs:  # Embedding
                op.inputs = [FIXED_INPUTS.inputs_embeds]
                embed_outputs = op.outputs[0]
                embed_op_id = op_id
                logger.info(f"Got Embedding op, embed_op_id: {embed_op_id}")
                continue
            if embed_outputs is None:
                continue
            if embed_outputs in op.inputs:
                op.inputs = [FIXED_INPUTS.inputs_embeds if ii == embed_outputs else ii for ii in op.inputs]

        if embed_op_id >= 0:
            logger.info(f"Remove op: {self.operations[embed_op_id]}")
            logger.info(f"Replace op inner names {embed_outputs} -> {FIXED_INPUTS.inputs_embeds}")
            self.operations.pop(embed_op_id)

    def _stack_operations(self):
        stacked_operations, cur_stack, pre_stack_id = [], [], -1
        for ii in self.operations:
            cur_stack_id = self.get_cur_repeat_block_idx(ii.op_name)
            if cur_stack_id == -1:  # Not a repeated block, could be in or out block
                if len(cur_stack) > 0:
                    stacked_operations.append(cur_stack)
                stacked_operations.append(ii)
                cur_stack = []
                continue
            if pre_stack_id != cur_stack_id:  # Changed to another block
                if len(cur_stack) > 0:
                    stacked_operations.append(cur_stack)
                cur_stack = [ii]
                pre_stack_id = cur_stack_id
            else:  # Repeat block
                cur_stack.append(ii)

        all_inputs = set([ii for op in self.operations for ii in op.inputs])
        base_graph_in_tensors = self.model_inputs + self.weight_names
        valid_base_graph_inputs = [ii for ii in base_graph_in_tensors if ii in all_inputs]

        all_outputs = set([ii for op in self.operations for ii in op.outputs])
        base_graph_out_tensors = [ii for ii in all_outputs if ii not in all_inputs]

        # record inputs and outputs for stacked GraphOperations
        stacked_inputs, stacked_outputs = [valid_base_graph_inputs], [base_graph_out_tensors]
        for ops in stacked_operations:
            if not isinstance(ops, list):
                continue
            cur_inputs, cur_outputs, inplace_outputs = [], [], []
            for op in ops:
                for ii in op.inputs:
                    cur_inputs.append(ii)
                for ii in op.outputs:
                    cur_outputs.append(ii)
                    if ii in op.inputs:
                        inplace_outputs.append(ii)
            cur_inputs_set = set(cur_inputs)
            cur_outputs_set = set(cur_outputs)
            inplace_outputs_set = set(inplace_outputs)
            stacked_inputs.append(list(cur_inputs_set - (cur_outputs_set - inplace_outputs_set)))
            stacked_outputs.append(list(cur_outputs_set & (all_inputs - cur_inputs_set)))
        return stacked_operations, stacked_inputs, stacked_outputs


def transform(source_path, input_names=BASIC_INPUT_NAMES, output_file=None, to_quant=False, quant_disable_names=None):
    logger.info("Building model using transformers...")
    config = AutoConfig.from_pretrained(source_path, local_files_only=True)
    mindie_model_file = find_mindie_supported_model(config)
    if mindie_model_file:
        from pathlib import Path

        contents_str = Path(__file__).with_name("run.py").read_text()
        run_pa_path = atb_speed_path + "/examples/models/{}/run_pa.sh".format(mindie_model_file)
        run_pa_path = check_and_get_real_path(run_pa_path, FileCheckConst.READ_ABLE,
                                              file_type=FileCheckConst.SHELL_SUFFIX, is_strict=True)
        contents_str = contents_str.replace('run_pa_path = "xxx"', f'run_pa_path = "{run_pa_path}"')
        contents_str = contents_str.replace("mindie_supported = False", "mindie_supported = True")
        contents_str = contents_str.replace('MODEL_PATH = "model_path_placeholder"', f'MODEL_PATH = "{source_path}"')
        infer_file = "run.py"
        write_file(infer_file, contents_str)
    else:
        model, config, is_vl_model = build_transformers_model(source_path)

        if is_vl_model:
            logger.info("Got VL model")

        logger.info("Transforming to atb")
        atb_model = ATBModelFromTorch(
            torch_model=model,
            config=config,
            input_names=input_names,
            to_quant=to_quant,
            quant_disable_names=quant_disable_names,
            is_vl_model=is_vl_model,
        )
        output_file = atb_model.to_file(output_file=output_file)

        logger.info("=" * 30)
        logger.info(f"Saved to: {output_file}\n")

        logger.info("=" * 30)
        logger.info(f"atb_model config:\n{atb_model.atb_model_config.to_dict()}\n")

        model_name = os.path.splitext(os.path.basename(output_file))[0]
        input_info = "input_ids=torch.arange(input_len)"
        if FIXED_INPUTS.position_ids in atb_model.inputs:
            input_info += ", position_ids=torch.arange(input_len)"
        logger.info("=" * 30)
        logger.info(
            f"""Run simple inference like:

        python3 -c "
        import torch, torch_npu
        import {model_name}
        from msit_llm.transform.torch_to_atb_python import ATBModel

        atb_model = ATBModel({model_name}.Model())
        weights = torch.load(\'$WEIGHT_PATH\')  # Use actual WEIGHT_PATH
        atb_model.set_weights(weights)

        input_len = 32
        out = atb_model.forward({input_info})
        print(out)
        "
        """.replace(
                " " * 4, ""
            )
        )

        infer_file = generate_infer_file(output_file, source_path, is_vl_model=is_vl_model)
        logger.info("=" * 30)
    logger.info(f"End-to-end inference example saved to: {infer_file}")
    logger.info(f"Execute by: python {infer_file}\n")
