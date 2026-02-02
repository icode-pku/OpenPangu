import os
import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from msit_llm.transform.torch_to_atb_python.utils import (
    get_config_attr,
    build_transformers_model,
    to_transformers_traced_module,
    get_lambda_source_code,
    get_valid_name,
    generate_infer_file,
    Operation,
    ATBModel,
    ATBModelConfig,
)
from msit_llm.transform.torch_to_atb_python.env import CONFIG_ATTR_CANDIDATES, VALID_NAME_CHARS


class TestUtils(unittest.TestCase):

    def test_generate_infer_file_given_invalid_path_when_os_error_then_raise_exception(self):
        with patch("pathlib.Path.read_text", side_effect=OSError("Error reading file")):
            with self.assertRaises(OSError):
                generate_infer_file("/invalid/path/output.atb", "/invalid/source", is_vl_model=False)

    @patch("msit_llm.transform.torch_to_atb_python.utils.write_file")
    @patch("pathlib.Path.read_text", return_value="model_path_placeholder")
    def test_generate_infer_file_given_valid_input_when_called_then_create_file(self, mock_read_text, mock_write_file):
        output_file = "/path/to/output_model.atb"
        source_path = "/path/to/source_model"

        result = generate_infer_file(output_file, source_path, is_vl_model=False)
        expected_path = Path("/path/to/run.py")
        self.assertEqual(result, expected_path)
        mock_write_file.assert_called_once()

    @patch("msit_llm.transform.torch_to_atb_python.utils.write_file")
    @patch("pathlib.Path.read_text", return_value="model_path_placeholder")
    def test_generate_infer_file_given_vl_model_when_called_then_create_vl_file(self, mock_read_text, mock_write_file):
        output_file = "/path/to/output_model.atb"
        source_path = "/path/to/source_model"

        result = generate_infer_file(output_file, source_path, is_vl_model=True)
        expected_path = Path("/path/to/run_vl.py")
        self.assertEqual(result, expected_path)
        mock_write_file.assert_called_once()

    def test_get_config_attr_given_valid_attr_when_in_candidates_then_return_sub_attr_value(self):
        config = MagicMock()
        config.sub_attr = "sub_value"
        CONFIG_ATTR_CANDIDATES["test_attr"] = ["sub_attr"]
        result = get_config_attr(config, "test_attr", default=None)
        self.assertEqual(result, "sub_value")

    def test_get_lambda_source_code_given_valid_function_when_called_then_return_source_code(self):
        def sample_lambda(): return x + 1  # Simple lambda function

        with patch("inspect.getsource", return_value="function=lambda x: x + 1, inputs=[]"):
            result = get_lambda_source_code(sample_lambda)
            self.assertEqual(result, "lambda x: x + 1")

    def test_get_valid_name_given_valid_string_when_all_chars_valid_then_return_same_string(self):
        result = get_valid_name("validName123")
        self.assertEqual(result, "validName123")

    def test_get_valid_name_given_valid_string_when_contains_invalid_chars_then_return_filtered_name(self):
        result = get_valid_name("valid@name!123")
        self.assertEqual(result, "validname123")

    def test_atbmodel_config_init_when_valid_parameters_then_set_correctly(self):
        config = ATBModelConfig(vocab_size=32000, num_attention_heads=12, max_batch_size=2)
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.max_batch_size, 2)

    def test_get_config_attr_given_config_when_attr_valid_then_return_sub_attr(self):
        mock_config = MagicMock()
        mock_config.text_config = "sub_config_value"
        config_candidate = {"text_config": ["text_config"]}
        with patch("msit_llm.transform.torch_to_atb_python.env.CONFIG_ATTR_CANDIDATES", config_candidate):
            result = get_config_attr(mock_config, "text_config")
            self.assertEqual(result, "sub_config_value")

    def test_get_valid_name_given_invalid_chars_then_remove_invalid_chars(self):
        result = get_valid_name("invalid@#name")
        self.assertEqual(result, "invalidname")

    def test_get_valid_name_given_valid_string_then_return_cleaned_name(self):
        result = get_valid_name("valid_name123")
        self.assertEqual(result, "valid_name123")

    def test_operation_copy_when_called_then_return_identical_copy(self):
        op = Operation(op_type="Mul", op_name="op2")
        op_copy = op.copy()
        self.assertEqual(op.to_dict(), op_copy.to_dict())
        self.assertIsNot(op, op_copy)

    def test_operation_to_dict_when_initialized_then_correct_dict(self):
        op = Operation(op_type="Add", op_name="op1", inputs=["x", "y"], outputs=["z"])
        expected_dict = {
            "op_type": "Add",
            "op_param": {},
            "inputs": ["x", "y"],
            "outputs": ["z"],
            "op_name": "op1",
            "function": None,
            "is_weights_first": False
        }
        self.assertEqual(op.to_dict(), expected_dict)

    def test_to_transformers_traced_module_given_model_when_valid_then_return_traced_module(self):
        mock_model = MagicMock()
        with patch("msit_llm.transform.torch_to_atb_python.utils.symbolic_trace") as mock_trace:
            mock_trace.return_value = "traced_module"
            result = to_transformers_traced_module(mock_model)
            self.assertEqual(result, "traced_module")

    def test_get_config_attr_given_config_attr_when_invalid_then_returned(self):
        config = ATBModelConfig(vocab_size=1, num_attention_heads=1, num_key_value_heads=1, head_dim=1, max_batch_size=1, max_seq_len=1024, rope_theta=1e4)
        attr = "non_existent_attr"
        default = 0
        result = get_config_attr(config, attr, default)
        self.assertEqual(result, 0)

    def test_get_config_attr_given_config_attr_when_valid_then_returned(self):
        config = ATBModelConfig(vocab_size=1, num_attention_heads=1, num_key_value_heads=1, head_dim=1, max_batch_size=1, max_seq_len=1024, rope_theta=1e4)
        attr = "vocab_size"
        default = 0
        result = get_config_attr(config, attr, default)
        self.assertEqual(result, 1)

    def test_operation_to_dict_with_invalidname1_when_initialized_then_correct_dict(self):
        op = Operation(op_type="Add", op_name="op@name", inputs=["x", "y"], outputs=["z"])
        expected_dict = {
            "op_type": "Add",
            "op_param": {},
            "inputs": ["x", "y"],
            "outputs": ["z"],
            "op_name": "op@name",
            "function": None,
            "is_weights_first": False
        }
        self.assertEqual(op.to_dict(), expected_dict)

    def test_operation_to_dict_with_invalidname2_when_initialized_then_correct_dict(self):
        op = Operation(op_type="Add", op_name="op!_name", inputs=["x", "y"], outputs=["z"])
        expected_dict = {
            "op_type": "Add",
            "op_param": {},
            "inputs": ["x", "y"],
            "outputs": ["z"],
            "op_name": "op!_name",
            "function": None,
            "is_weights_first": False
        }
        self.assertEqual(op.to_dict(), expected_dict)

    def test_get_valid_name_given_name_when_invalid_then_returned(self):
        name = "Invalid Name!@#"
        result = get_valid_name(name)
        self.assertEqual(result, "InvalidName")

    def test_get_valid_name_given_name_when_valid_then_returned(self):
        name = "ValidName123"
        result = get_valid_name(name)
        self.assertEqual(result, "ValidName123")


if __name__ == "__main__":
    unittest.main()