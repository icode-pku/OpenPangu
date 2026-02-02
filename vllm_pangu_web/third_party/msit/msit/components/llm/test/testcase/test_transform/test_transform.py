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

import json
from unittest import TestCase
from unittest.mock import Mock

from msit_llm.transform.torch_to_atb_python.utils import get_config_attr, get_valid_name, Operation, ATBModelConfig
from msit_llm.transform.torch_to_atb_python.env import NN_MODULE_STACK, FX_OP_TYPES
from msit_llm.transform.torch_to_atb_python.torch_to_atb_python import ATBModelFromTorch


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestGetConfigAttr(TestCase):
    def setUp(self):
        self.config1 = Config(
            num_hidden_layers=4,
            num_attention_heads=8,
            hidden_size=512,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            vocab_size=30522,
            llm_config="config1",
        )
        self.config2 = Config(
            num_layers=4,
            num_attention_heads=8,
            hidden_size=512,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            vocab_size=30522,
            llm_config="config2",
        )
        self.config3 = Config(
            n_layers=4,
            num_attention_heads=8,
            hidden_size=512,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            vocab_size=30522,
            text_config="config3",
        )

    def test_get_existing_attributes(self):
        self.assertEqual(get_config_attr(self.config1, "num_hidden_layers"), 4)
        self.assertEqual(get_config_attr(self.config1, "num_attention_heads"), 8)
        self.assertEqual(get_config_attr(self.config1, "hidden_size"), 512)
        self.assertEqual(get_config_attr(self.config1, "rms_norm_eps"), 1e-6)
        self.assertEqual(get_config_attr(self.config1, "rope_theta"), 10000.0)
        self.assertEqual(get_config_attr(self.config1, "vocab_size"), 30522)
        self.assertEqual(get_config_attr(self.config1, "text_config"), "config1")

    def test_get_candidate_attributes(self):
        self.assertEqual(get_config_attr(self.config2, "num_hidden_layers"), 4)
        self.assertEqual(get_config_attr(self.config3, "num_hidden_layers"), 4)
        self.assertEqual(get_config_attr(self.config2, "text_config"), "config2")
        self.assertEqual(get_config_attr(self.config3, "text_config"), "config3")

    def test_get_nonexistent_attribute_with_default(self):
        self.assertEqual(get_config_attr(self.config1, "nonexistent", "default_value"), "default_value")
        self.assertEqual(get_config_attr(self.config1, "nonexistent"), None)


class TestGetValidName(TestCase):
    def test_empty_string(self):
        self.assertEqual(get_valid_name(""), "")

    def test_all_valid_chars(self):
        self.assertEqual(get_valid_name("valid_name123"), "valid_name123")
        self.assertEqual(get_valid_name("another_valid_name_456"), "another_valid_name_456")

    def test_mixed_chars(self):
        self.assertEqual(get_valid_name("invalid_name123"), "invalid_name123")
        self.assertEqual(get_valid_name("anoth3r_valid_nam3.789"), "anoth3r_valid_nam3.789")

    def test_only_invalid_chars(self):
        self.assertEqual(get_valid_name("!@#$%^&*()"), "")
        self.assertEqual(get_valid_name("[]{}<>;:'\"|,?/\\ "), "")

    def test_unicode_chars(self):
        self.assertEqual(get_valid_name("ünicöde_tést"), "nicde_tst")
        self.assertEqual(get_valid_name("测试_测试"), "_")


class TestOperation(TestCase):
    def setUp(self):
        self.op1 = Operation(
            op_type="attention",
            op_param={"alpha": 1.0},
            inputs=["input_1", "input_2"],
            outputs=["output"],
            op_name="attention_op",
            function=lambda a, b: a + b,
            is_weights_first=True,
        )

    def test_init(self):
        self.assertEqual(self.op1.op_type, "attention")
        self.assertEqual(self.op1.op_param, {"alpha": 1.0})
        self.assertEqual(self.op1.inputs, ["input_1", "input_2"])
        self.assertEqual(self.op1.outputs, ["output"])
        self.assertEqual(self.op1.op_name, "attention_op")
        self.assertIsNotNone(self.op1.function)
        self.assertTrue(self.op1.is_weights_first)

    def test_to_dict(self):
        # test to_dict
        expected_dict = {
            "op_type": "attention",
            "op_param": {"alpha": 1.0},
            "inputs": ["input_1", "input_2"],
            "outputs": ["output"],
            "op_name": "attention_op",
            "function": self.op1.function,
            "is_weights_first": True,
        }
        self.assertEqual(self.op1.to_dict(), expected_dict)

    def test_to_json(self):
        # test to_json
        expected_json = {
            "op_type": "attention",
            "inputs": ["input_1", "input_2"],
            "outputs": ["output"],
            "op_name": "attention_op",
            "function": "lambda a, b: a + b",
        }
        self.assertEqual(self.op1.to_json(), expected_json)

    def test_copy(self):
        # test copy
        op1_copy = self.op1.copy()
        self.assertIsNot(op1_copy, self.op1)
        self.assertEqual(op1_copy.to_dict(), self.op1.to_dict())


class TestATBModelConfig(TestCase):
    def setUp(self):
        self.config1 = ATBModelConfig(
            vocab_size=30522,
            num_attention_heads=12,
            num_key_value_heads=8,
            head_dim=64,
            max_batch_size=16,
            max_seq_len=512,
            rope_theta=10000.0,
            custom_param1="value1",
            custom_param2=42,
        )

        self.config2 = ATBModelConfig(
            vocab_size=49152,
            num_attention_heads=16,
            num_key_value_heads=16,
            head_dim=768,
            rope_theta=10000.0,
            max_batch_size=32,
            max_seq_len=1024,
            custom_param3=True,
        )

    def test_init(self):
        self.assertEqual(self.config1.vocab_size, 30522)
        self.assertEqual(self.config1.num_attention_heads, 12)
        self.assertEqual(self.config1.num_key_value_heads, 8)
        self.assertEqual(self.config1.head_dim, 64)
        self.assertEqual(self.config1.max_batch_size, 16)
        self.assertEqual(self.config1.max_seq_len, 512)
        self.assertEqual(self.config1.rope_theta, 10000.0)
        self.assertEqual(self.config1.custom_param1, "value1")
        self.assertEqual(self.config1.custom_param2, 42)

    def test_to_dict(self):
        # test to_dict
        expected_dict1 = {
            "vocab_size": 30522,
            "num_attention_heads": 12,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "rope_theta": 10000.0,
            "max_batch_size": 16,
            "max_seq_len": 512,
            "custom_param1": "value1",
            "custom_param2": 42,
        }
        self.assertEqual(self.config1.to_dict(), expected_dict1)

        expected_dict2 = {
            "vocab_size": 49152,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 768,
            "rope_theta": 10000.0,
            "max_batch_size": 32,
            "max_seq_len": 1024,
            "custom_param3": True,
        }
        self.assertEqual(self.config2.to_dict(), expected_dict2)

    def test_repr(self):
        expected_json1 = json.dumps(
            {
                "vocab_size": 30522,
                "num_attention_heads": 12,
                "num_key_value_heads": 8,
                "head_dim": 64,
                "rope_theta": 10000.0,
                "max_batch_size": 16,
                "max_seq_len": 512,
                "custom_param1": "value1",
                "custom_param2": 42,
            }
        )
        self.assertEqual(repr(self.config1), expected_json1)
        expected_json2 = json.dumps(
            {
                "vocab_size": 49152,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "head_dim": 768,
                "rope_theta": 10000.0,
                "max_batch_size": 32,
                "max_seq_len": 1024,
                "custom_param3": True,
            }
        )
        self.assertEqual(repr(self.config2), expected_json2)


class TestATBModelFromTorch(TestCase):
    def test_get_cur_repeat_block_idx(self):
        #  test get_cur_repeat_block_idx
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("encoder.0"), 0)
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("decoder.1"), 1)
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("layer.2"), 2)
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("transformer.3"), 3)
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("block.4"), 4)
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("attention.5"), 5)
        # without number
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("encoder"), -1)
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("decoder"), -1)
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("layer"), -1)
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("transformer"), -1)
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("block"), -1)
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("attention"), -1)
        # numbers
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("encoder.0.1.2"), 0)
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("decoder.1.2.3"), 1)
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("layer.2.3.4"), 2)
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("transformer.3.4.5"), 3)
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("block.4.5.6"), 4)

    def test_no_digits(self):
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("module.submodule"), -1)

    def test_empty_string(self):
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx(""), -1)

    def test_non_ascii_characters(self):
        self.assertEqual(ATBModelFromTorch.get_cur_repeat_block_idx("módulo.submódulo"), -1)

    def test_get_module_name_by_nn_module_stack(self):
        nn_module_stack = {"module1": "Linear", "module2": "ReLU", "module3": "Dropout"}
        node = Mock()
        node.meta = {NN_MODULE_STACK: nn_module_stack}
        self.assertEqual(ATBModelFromTorch._get_module_name_by_nn_module_stack(node), "module3")
        single_module_stack = {"module1": "Linear"}
        node.meta = {NN_MODULE_STACK: single_module_stack}
        self.assertEqual(ATBModelFromTorch._get_module_name_by_nn_module_stack(node), "module1")
        empty_module_stack = {}
        node.meta = {NN_MODULE_STACK: empty_module_stack}
        self.assertIsNone(ATBModelFromTorch._get_module_name_by_nn_module_stack(node))

    def test_should_skip_node(self):
        node = Mock()
        # test call_method
        node.op = FX_OP_TYPES.call_method
        node.target = "size"
        self.assertTrue(ATBModelFromTorch._should_skip_node(node))
        node.target = "view"
        self.assertFalse(ATBModelFromTorch._should_skip_node(node))
        # test call_function
        node.op = FX_OP_TYPES.call_function
        node.target = Mock(__name__="dropout")
        self.assertTrue(ATBModelFromTorch._should_skip_node(node))
        # test is_wrapped
        node.meta = {"is_wrapped": True}
        self.assertTrue(ATBModelFromTorch._should_skip_node(node))
        # test other node
        node.op = FX_OP_TYPES.placeholder
        node.target = "input"
        self.assertFalse(ATBModelFromTorch._should_skip_node(node))
        node.op = FX_OP_TYPES.output
        node.target = "output"
        self.assertFalse(ATBModelFromTorch._should_skip_node(node))
        node.op = FX_OP_TYPES.get_attr
        node.target = "attr"
        self.assertFalse(ATBModelFromTorch._should_skip_node(node))
