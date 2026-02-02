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

import unittest
import base64
from unittest.mock import patch
import onnx
import pandas as pd
from io import StringIO

from msit_llm.common.json_fitter import atb_node_to_plain_node, atb_json_dict_node_parse, \
    atb_param_to_onnx_attribute, atb_param_to_onnx_attribute, parse_onnx_attr_from_atb_node_dict, \
    atb_node_to_onnx_node, build_onnx_shape_info, TYPESTR2ONNXTYPE, atb_shape_to_onnx_shape, \
    csv_to_content, atb_json_to_onnx, atb_json_to_onnx_json


class TestAtbNodeToPlainNode(unittest.TestCase):
    def test_target_level_equal_current_level(self):
        node = {"name": "root", "level": 0}
        result = atb_node_to_plain_node(node, 0, 0)
        self.assertEqual(result, [node])

    def test_without_child_nodes(self):
        node = {"name": "leaf"}
        result = atb_node_to_plain_node(node, 0, -1)
        self.assertEqual(result, [node])

    def test_target_level_greater_than_current_level(self):
        node = {
            "name": "root",
            "nodes": [
                {"name": "child1"},
                {"name": "child2", "nodes": [{"name": "grandchild"}]}
            ]
        }
        expected_result = [{'name': 'child1'}, {'name': 'child2', 'nodes': [{'name': 'grandchild'}]}]
        result = atb_node_to_plain_node(node, 0, 1)
        self.assertEqual(result, expected_result)

    def test_target_level_minus_one(self):
        node = {
            "name": "root",
            "nodes": [
                {"name": "child1"},
                {"name": "child2", "nodes": [{"name": "grandchild"}]}
            ]
        }
        expected_result = [{'name': 'child1'}, {'name': 'grandchild'}]
        result = atb_node_to_plain_node(node, 0, -1)
        self.assertEqual(result, expected_result)


class TestAtbJsonDictNodeParse(unittest.TestCase):

    def test_target_level_zero(self):
        # Test  target_level is 0
        atb_json_dict = {"name": "root", "nodes": [{"name": "child"}]}
        result = atb_json_dict_node_parse(atb_json_dict, 0)
        self.assertEqual(result, [atb_json_dict])

    def test_no_child_nodes(self):
        # Test without child nodes
        atb_json_dict = {"name": "root"}
        result = atb_json_dict_node_parse(atb_json_dict, 1)
        self.assertEqual(result, [atb_json_dict])

    def test_with_child_nodes(self):
        atb_json_dict = {
            "name": "root",
            "nodes": [
                {"name": "child1"},
                {"name": "child2", "nodes": [{"name": "grandchild"}]}
            ]
        }
        expected_result = [
            {'name': 'child1'}, 
            {'name': 'grandchild'}
        ]
        result = atb_json_dict_node_parse(atb_json_dict, -1)
        self.assertEqual(result, expected_result)

    def test_specific_target_level(self):
        # Test the situation at a specific target level
        atb_json_dict = {
            "name": "root",
            "nodes": [
                {"name": "child1"},
                {"name": "child2", "nodes": 
                    [{"name": "grandchild"}]
                }
            ]
        }
        expected_result = [{'name': 'child1'}, {'name': 'child2', 'nodes': [{'name': 'grandchild'}]}]
        result = atb_json_dict_node_parse(atb_json_dict, 1)
        self.assertEqual(result, expected_result)

    def test_nested_structure(self):
        atb_json_dict = {
            "name": "root",
            "nodes": [
                {"name": "child1", "nodes": [{"name": "subchild1"}]},
                {"name": "child2", "nodes": [
                    {"name": "subchild2", "nodes": [{"name": "subsubchild"}]}
                ]}
            ]
        }
        expected_result = [{'name': 'subchild1'}, {'name': 'subsubchild'}]
        result = atb_json_dict_node_parse(atb_json_dict, -1)
        self.assertEqual(result, expected_result)


class TestAtbParamToOnnxAttribute(unittest.TestCase):
    def test_string_type(self):
        # Test parameters of type string
        param_name = "test_param"
        param_value = "hello world"
        expected_result = {
            "name": param_name,
            "type": "STRINGS",
            "strings": [base64.b64encode(param_value.encode("utf-8")).decode("utf-8")]
        }
        result = atb_param_to_onnx_attribute(param_name, param_value)
        self.assertEqual(result, expected_result)

    @patch('msit_llm.common.json_fitter.logger.debug')
    def test_unicode_encode_error(self, mock_debug):
        # Testing unencodable strings
        param_name = "test_param"
        param_value = "invalid \ud800 string"  # Contains invalid Unicode characters
        expected_result = {
            "name": param_name,
            "type": "STRINGS"
        }
        result = atb_param_to_onnx_attribute(param_name, param_value)
        self.assertEqual(result, expected_result)
        mock_debug.assert_called()

    @patch('msit_llm.common.json_fitter.logger.debug')
    def test_unicode_decode_error(self, mock_debug):
        # Testing undecoded strings
        param_name = "test_param"
        param_value = "\xff\xff"  
        expected_result = {
            "name": param_name,
            "type": "STRINGS"
        }
        with patch('base64.b64encode', side_effect=UnicodeDecodeError('utf-8', b'\xff\xff', 0, 2, 'Invalid data')):
            result = atb_param_to_onnx_attribute(param_name, param_value)
        self.assertEqual(result, expected_result)
        mock_debug.assert_called()

    def test_float_list_type(self):
        param_name = "test_param"
        param_value = [1.0, 2.5, 3.7]
        expected_result = {
            "name": param_name,
            "type": "FLOATS",
            "floats": param_value
        }
        result = atb_param_to_onnx_attribute(param_name, param_value)
        self.assertEqual(result, expected_result)

    def test_single_float_type(self):
        param_name = "test_param"
        param_value = 42.0
        expected_result = {
            "name": param_name,
            "type": "FLOATS",
            "floats": [param_value]
        }
        result = atb_param_to_onnx_attribute(param_name, param_value)
        self.assertEqual(result, expected_result)

    def test_nonexistent_value(self):
        param_name = "test_param"
        param_value = None
        expected_result = {
            "name": param_name,
            "type": "FLOATS",
            "floats": []
        }
        result = atb_param_to_onnx_attribute(param_name, param_value)
        self.assertEqual(result, expected_result)


class TestParseOnnxAttrFromAtbNodeDict(unittest.TestCase):

    def test_no_param(self):
        node_dict = {"name": "node_without_param"}
        result = parse_onnx_attr_from_atb_node_dict(node_dict)
        self.assertEqual(result, [])

    def test_simple_params(self):
        node_dict = {
            "name": "node_with_simple_params",
            "param": {
                "param1": "value1",
                "param2": 42.0
            }
        }
        expected_result = [
            atb_param_to_onnx_attribute("param1", "value1"),
            atb_param_to_onnx_attribute("param2", 42.0)
        ]
        result = parse_onnx_attr_from_atb_node_dict(node_dict)
        self.assertEqual(result, expected_result)

    def test_dict_params(self):
        node_dict = {
            "name": "node_with_dict_params",
            "param": {
                "dict_param": {
                    "sub_param1": "sub_value1",
                    "sub_param2": 5.5
                }
            }
        }
        expected_result = [
            atb_param_to_onnx_attribute("dict_param.sub_param1", "sub_value1"),
            atb_param_to_onnx_attribute("dict_param.sub_param2", 5.5),
            atb_param_to_onnx_attribute("dict_param.sub_param2", 5.5)
        ]

        result = parse_onnx_attr_from_atb_node_dict(node_dict)
        self.assertEqual(result, expected_result)

    def test_mixed_params(self):
        node_dict = {
            "name": "node_with_mixed_params",
            "param": {
                "simple_param": "simple_value",
                "dict_param": {
                    "sub_param1": "sub_value1",
                    "sub_param2": 5.5
                },
                "another_simple_param": 100.0
            }
        }
        expected_result = [
            atb_param_to_onnx_attribute("simple_param", "simple_value"),
            atb_param_to_onnx_attribute("dict_param.sub_param1", "sub_value1"),
            atb_param_to_onnx_attribute("dict_param.sub_param2", 5.5),
            atb_param_to_onnx_attribute("dict_param.sub_param2", 5.5),
            atb_param_to_onnx_attribute("another_simple_param", 100.0)
        ]
        result = parse_onnx_attr_from_atb_node_dict(node_dict)
        self.assertEqual(result, expected_result)

    def test_none_param_value(self):
        node_dict = {
            "name": "node_with_none_param",
            "param": {
                "none_param": None,
                "valid_param": "valid_value"
            }
        }
        expected_result = [
            atb_param_to_onnx_attribute("none_param", None),
            atb_param_to_onnx_attribute("valid_param", "valid_value")
        ]
        result = parse_onnx_attr_from_atb_node_dict(node_dict)
        self.assertEqual(result, expected_result)


class TestAtbNodeToOnnxNode(unittest.TestCase):
    def test_standard_conversion(self):
        atb_node_dict = {
            "opName": "node_name",
            "opType": "Conv",
            "inTensors": ["input_tensor"],
            "outTensors": ["output_tensor"],
            "param": {
                "param1": "value1",
                "dict_param": {
                    "sub_param1": "sub_value1"
                }
            }
        }
        expected_result = {
            "name": "node_name",
            "opType": "Conv",
            "input": ["input_tensor"],
            "output": ["output_tensor"],
            "attribute": [
                parse_onnx_attr_from_atb_node_dict(
                    {"param": {"param1": "value1", "dict_param": {"sub_param1": "sub_value1"}}})[0],
                parse_onnx_attr_from_atb_node_dict(
                    {"param": {"dict_param": {"sub_param1": "sub_value1"}}})[0],
                parse_onnx_attr_from_atb_node_dict(
                    {"param": {"dict_param": {"sub_param1": "sub_value1"}}})[0]
            ]
        }
        result = atb_node_to_onnx_node(atb_node_dict)
        self.assertEqual(result["name"], expected_result["name"])
        self.assertEqual(result["opType"], expected_result["opType"])
        self.assertEqual(result["input"], expected_result["input"])
        self.assertEqual(result["output"], expected_result["output"])
        self.assertEqual(len(result["attribute"]), len(expected_result["attribute"]))
        for attr in result["attribute"]:
            self.assertIn(attr, expected_result["attribute"])

    def test_empty_param(self):
        atb_node_dict = {
            "opName": "node_name",
            "opType": "Conv",
            "inTensors": ["input_tensor"],
            "outTensors": ["output_tensor"]
        }
        expected_result = {
            "name": "node_name",
            "opType": "Conv",
            "input": ["input_tensor"],
            "output": ["output_tensor"],
            "attribute": []
        }
        result = atb_node_to_onnx_node(atb_node_dict)
        self.assertEqual(result, expected_result)

    def test_complex_param_structure(self):
        atb_node_dict = {
            "opName": "node_name",
            "opType": "ComplexOp",
            "inTensors": ["input_tensor"],
            "outTensors": ["output_tensor"],
            "param": {
                "simple_param": "simple_value",
                "dict_param": {
                    "sub_param1": "sub_value1",
                    "sub_param2": 5.5
                },
                "list_param": [1.0, 2.5, 3.7]
            }
        }
        expected_result = {
            "name": "node_name",
            "opType": "ComplexOp",
            "input": ["input_tensor"],
            "output": ["output_tensor"],
            "attribute": [
                parse_onnx_attr_from_atb_node_dict(
                    {"param": {"simple_param": "simple_value"}})[0],
                parse_onnx_attr_from_atb_node_dict(
                    {"param": {"dict_param": {"sub_param1": "sub_value1", "sub_param4": 5.5}}})[0],
                {'name': 'dict_param.sub_param2', 'type': 'FLOATS', 'floats': [5.5]},
                parse_onnx_attr_from_atb_node_dict(
                    {"param": {"dict_param": {"sub_param3": "sub_value1", "sub_param6": 5.5}}})[0],
                parse_onnx_attr_from_atb_node_dict(
                    {"param": {"list_param": [1.0, 2.5, 3.7]}})[0],
            ]
        }
        result = atb_node_to_onnx_node(atb_node_dict)
        self.assertEqual(result["name"], expected_result["name"])
        self.assertEqual(result["opType"], expected_result["opType"])
        self.assertEqual(result["input"], expected_result["input"])
        self.assertEqual(result["output"], expected_result["output"])
        self.assertEqual(len(result["attribute"]), len(expected_result["attribute"]))
        for attr in result["attribute"]:
            self.assertIn(attr, expected_result["attribute"])

    def test_empty_io_tensors(self):
        atb_node_dict = {
            "opName": "node_name",
            "opType": "Conv",
            "inTensors": [],
            "outTensors": [],
            "param": {
                "param1": "value1"
            }
        }
        expected_result = {
            "name": "node_name",
            "opType": "Conv",
            "input": [],
            "output": [],
            "attribute": [
                parse_onnx_attr_from_atb_node_dict({"param": {"param1": "value1"}})[0]
            ]
        }
        result = atb_node_to_onnx_node(atb_node_dict)
        self.assertEqual(result, expected_result)


class TestBuildOnnxShapeInfo(unittest.TestCase):
    def test_missing_input_shape_info(self):
        result = build_onnx_shape_info("test_tensor")
        expected_result = {"name": "test_tensor"}
        self.assertEqual(result, expected_result)

    def test_standard_conversion(self):
        input_shape_info = {
            "type": "float32",
            "shape": [1, 3, 224, 224]
        }
        result = build_onnx_shape_info("input_tensor", input_shape_info)
        expected_result = {
            "name": "input_tensor",
            "type": {
                "tensorType": {
                    "elemType": TYPESTR2ONNXTYPE["float32"],
                    "shape": {
                        "dim": [
                            {"dimValue": 1},
                            {"dimValue": 3},
                            {"dimValue": 224},
                            {"dimValue": 224}
                        ]
                    }
                }
            }
        }

        self.assertEqual(result, expected_result)

    def test_empty_shape(self):
        input_shape_info = {
            "type": "float32",
            "shape": []
        }
        result = build_onnx_shape_info("input_tensor", input_shape_info)
        expected_result = {
            "name": "input_tensor",
            "type": {
                "tensorType": {
                    "elemType": TYPESTR2ONNXTYPE["float32"],
                    "shape": {
                        "dim": []
                    }
                }
            }
        }
        self.assertEqual(result, expected_result)


class TestAtbShapeToOnnxShape(unittest.TestCase):
    def test_standard_conversion(self):
        value_info = []
        input_names = ["input_tensor_1", "input_tensor_2"]
        input_shapes = [
            {"type": "float32", "shape": [1, 3, 224, 224]},
            {"type": "int32", "shape": [2, 256]}
        ]
        expected_value_info = [
            build_onnx_shape_info("input_tensor_1", input_shapes[0]),
            build_onnx_shape_info("input_tensor_2", input_shapes[1])
        ]
        atb_shape_to_onnx_shape(value_info, input_names, input_shapes)
        self.assertEqual(value_info, expected_value_info)

    def test_mismatched_input_lengths(self):
        value_info = []
        input_names = ["input_tensor_1", "input_tensor_2", "input_tensor_3"]
        input_shapes = [
            {"type": "float32", "shape": [1, 3, 224, 224]},
            {"type": "int32", "shape": [2, 256]}
        ]
        expected_value_info = [
            build_onnx_shape_info("input_tensor_1", input_shapes[0]),
            build_onnx_shape_info("input_tensor_2", input_shapes[1])
        ]
        
        atb_shape_to_onnx_shape(value_info, input_names, input_shapes)
        self.assertEqual(value_info, expected_value_info)

    def test_empty_inputs(self):
        value_info = []
        input_names = []
        input_shapes = []
        expected_value_info = []
        atb_shape_to_onnx_shape(value_info, input_names, input_shapes)
        self.assertEqual(value_info, expected_value_info)

    def test_empty_shapes(self):
        value_info = []
        input_names = ["input_tensor_1"]
        input_shapes = [{"type": "float32", "shape": []}]
        expected_value_info = [
            build_onnx_shape_info("input_tensor_1", input_shapes[0])
        ]
        
        atb_shape_to_onnx_shape(value_info, input_names, input_shapes)
        self.assertEqual(value_info, expected_value_info)

class IdentityFunction:
    @staticmethod
    def __call__(self, x):
        return x


class TestAtbFunctions(unittest.TestCase):

    def test_atb_node_to_plain_node(self):
        atb_node_dict = {
            "opName": "test_node",
            "opType": "TestOp",
            "nodes": [
                {
                    "opName": "child_node_1",
                    "opType": "ChildOp",
                    "nodes": []
                },
                {
                    "opName": "child_node_2",
                    "opType": "ChildOp",
                    "nodes": []
                }
            ]
        }
        result = atb_node_to_plain_node(atb_node_dict, 0, -1)
        self.assertEqual(len(result), 0)

    def test_atb_json_dict_node_parse(self):
        atb_json_dict = {
            "nodes": [
                {
                    "opName": "node_1",
                    "opType": "Op1",
                    "nodes": []
                },
                {
                    "opName": "node_2",
                    "opType": "Op2",
                    "nodes": []
                }
            ]
        }
        result = atb_json_dict_node_parse(atb_json_dict, 1)
        self.assertEqual(len(result), 2)  # 应该返回 2 个节点

    def test_atb_param_to_onnx_attribute(self):
        param_name = "test_param"
        param_value = "test_value"
        result = atb_param_to_onnx_attribute(param_name, param_value)
        self.assertEqual(result["name"], param_name)
        self.assertEqual(result["type"], "STRINGS")
        self.assertEqual(base64.b64decode(result["strings"][0]).decode("utf-8"), param_value)

    def test_parse_onnx_attr_from_atb_node_dict(self):
        atb_node_dict = {
            "param": {
                "param1": 1.0,
                "param2": {
                    "sub_param1": 2.0,
                    "sub_param2": 3.0
                }
            }
        }
        result = parse_onnx_attr_from_atb_node_dict(atb_node_dict)
        self.assertEqual(len(result), 4)

    def test_atb_node_to_onnx_node(self):
        atb_node_dict = {
            "opName": "test_op",
            "opType": "TestOp",
            "inTensors": ["input1", "input2"],
            "outTensors": ["output1"],
            "param": {
                "param1": 1.0
            }
        }
        result = atb_node_to_onnx_node(atb_node_dict)
        self.assertEqual(result["name"], "test_op")
        self.assertEqual(result["opType"], "TestOp")
        self.assertEqual(result["input"], ["input1", "input2"])
        self.assertEqual(result["output"], ["output1"])
        self.assertEqual(len(result["attribute"]), 1)  # 应该返回 1 个属性

    def test_build_onnx_shape_info(self):
        name = "input_tensor"
        input_shape_info = {
            "type": "float32",
            "shape": [1, 3, 224, 224]
        }
        result = build_onnx_shape_info(name, input_shape_info)
        self.assertEqual(result["name"], name)
        self.assertEqual(result["type"]["tensorType"]["elemType"], onnx.helper.TensorProto.FLOAT)
        self.assertEqual(result["type"]["tensorType"]["shape"]["dim"][0]["dimValue"], 1)

    def test_atb_shape_to_onnx_shape(self):
        value_info = []
        input_names = ["input1", "input2"]
        input_shapes = [{"type": "float32", "shape": [1, 2]}, {"type": "int32", "shape": [3]}]

        atb_shape_to_onnx_shape(value_info, input_names, input_shapes)

        expected_value_info = [
            {
                "name": "input1",
                "type": {
                    "tensorType": {
                        "elemType": onnx.helper.TensorProto.FLOAT,
                        "shape": {"dim": [{"dimValue": 1}, {"dimValue": 2}]}
                    }
                }
            },
            {
                "name": "input2",
                "type": {
                    "tensorType": {
                        "elemType": onnx.helper.TensorProto.INT32,
                        "shape": {"dim": [{"dimValue": 3}]}
                    }
                }
            }
        ]

        self.assertEqual(value_info, expected_value_info)


class TestCsvToContent:
    """测试CSV内容转换函数"""
    
    # 基本CSV内容
    BASE_CSV = """OpName|InDType|InShape|OutDType|OutShape
conv1|float32|1,224,224,3|float32|1,112,112,32
pool1|float32|1,112,112,32|float32|1,56,56,32
dense1|float32|1,56,56,32;float32|1,100352|float32|1,512
"""
    
    @patch("pandas.read_csv")
    @patch("msit_llm.common.json_fitter.Rule")
    @patch("msit_llm.common.json_fitter.load_file_to_read_common_check")
    def test_conversion(self, mock_load, mock_rule, mock_read_csv):
        mock_load.return_value = "/path/to/op_info.csv"
        mock_rule.return_value.check.return_value = True
        
        mock_read_csv.return_value = pd.read_csv(
            StringIO(self.BASE_CSV), 
            sep="|", 
            dtype=str
        )
        
        # 执行测试
        result = csv_to_content("dummy_path")
        
        # 验证结果
        assert result == {}
        


class TestAtbToOnnxJson:
    """测试ATB JSON到ONNX JSON的转换函数"""
    
    # 基础ATB JSON结构
    BASE_ATB_JSON = {
        "inTensors": ["input_tensor"],
        "outTensors": ["output_tensor"],
        "nodes": [
            {"type": "Conv", "name": "conv1", "inputs": ["input_tensor"], "outputs": ["conv_output"]},
            {"type": "Relu", "name": "relu1", "inputs": ["conv_output"], "outputs": ["output_tensor"]}
        ]
    }
    
    # 基础形状信息
    BASE_SHAPE_CONTENTS = {
        "conv1": {
            "inputs": [{"type": "float32", "shape": ["1", "224", "224", "3"]}],
            "outputs": [{"type": "float32", "shape": ["1", "112", "112", "32"]}]
        },
        "relu1": {
            "inputs": [{"type": "float32", "shape": ["1", "112", "112", "32"]}],
            "outputs": [{"type": "float32", "shape": ["1", "112", "112", "32"]}]
        }
    }
    
    def mock_atb_json_dict_node_parse(atb_json_dict, target_level):
        """模拟节点解析函数"""
        return atb_json_dict["nodes"]
    
    def mock_atb_node_to_onnx_node(node):
        """模拟节点转换函数"""
        # 简化转换：保持相同结构但添加ONNX所需字段
        return {
            "name": node["name"],
            "op_type": node["type"],
            "input": node.get("inputs", []),
            "output": node.get("outputs", [])
        }
    
    def mock_atb_shape_to_onnx_shape(value_info_list, names, shape_info_list):
        """模拟形状转换函数"""
        for name, shape_info in zip(names, shape_info_list):
            value_info_list.append({
                "name": name,
                "type": shape_info["type"],
                "shape": shape_info["shape"]
            })
    
    @patch("msit_llm.common.json_fitter.atb_shape_to_onnx_shape", side_effect=mock_atb_shape_to_onnx_shape)
    @patch("msit_llm.common.json_fitter.atb_node_to_onnx_node", side_effect=mock_atb_node_to_onnx_node)
    @patch("msit_llm.common.json_fitter.atb_json_dict_node_parse", side_effect=mock_atb_json_dict_node_parse)
    def test_basic_conversion(self, mock_node_parse, mock_node_convert, mock_shape_convert):
        """测试基本转换功能"""
        # 执行测试
        result = atb_json_to_onnx_json(
            atb_json_dict=self.BASE_ATB_JSON, 
            target_level="level1", 
            shape_contents=self.BASE_SHAPE_CONTENTS
        )
        
        # 验证结果结构
        assert "graph" in result
        assert "node" in result["graph"]
        assert len(result["graph"]["node"]) == 2
        
        # 验证输入输出
        assert len(result["graph"]["input"]) == 1
        assert result["graph"]["input"][0]["name"] == "input_tensor"
        assert len(result["graph"]["output"]) == 1
        assert result["graph"]["output"][0]["name"] == "output_tensor"
        
        # 验证valueInfo
        assert "valueInfo" in result["graph"]
        assert len(result["graph"]["valueInfo"]) == 4  # 2节点的输入输出
        
        # 验证辅助函数调用
        mock_node_parse.assert_called_once_with(self.BASE_ATB_JSON, "level1")
        assert mock_node_convert.call_count == 2
        assert mock_shape_convert.call_count == 4  # 每个节点调用两次（输入和输出）