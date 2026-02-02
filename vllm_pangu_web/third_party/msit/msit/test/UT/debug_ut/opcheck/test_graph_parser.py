import unittest
import json
from unittest.mock import patch, mock_open
from msit_opcheck.graph_parser import get_single_op_info_from_op_list, get_ge_graph_name, get_all_opinfo
from msit_opcheck.graph_parser import InputOutputDesc, OpInfo


class TestGetSingleOpInfoFromOpList(unittest.TestCase):
    def test_op_with_slash_in_name(self):
        """测试操作名称包含斜杠的情况"""
        op_list = [{"name": "module/conv1", "type": "Conv2D"}]
        result = get_single_op_info_from_op_list(op_list, {}, {})
        
        self.assertIn("module_conv1", result)

    def test_empty_op_list(self):
        """测试空操作列表"""
        result = get_single_op_info_from_op_list([], {}, {})
        self.assertEqual(len(result), 0)

    def test_op_without_name(self):
        """测试操作没有name字段的情况"""
        op_list = [
            {"type": "Conv2D"},
            {"name": "pool1", "type": "MaxPool"}
        ]
        result = get_single_op_info_from_op_list(op_list, {}, {})
        self.assertEqual(len(result), 1)
        self.assertIn("pool1", result)


class TestInputOutputDescInit(unittest.TestCase):
    def test_init_with_all_attributes(self):
        """Test initialization with all possible attributes."""
        kwargs = {
            "attr": ["some_attr"],
            "device_type": "NPU",
            "dtype": "DT_BOOL"
        }
        obj = InputOutputDesc(**kwargs)
        self.assertEqual(obj.attr, ["some_attr"])
        self.assertEqual(obj.input_output_param, kwargs)


class TestGetGeGraphName(unittest.TestCase):

    def test_normal_case_with_multiple_graphs(self):
        valid_json_content = {
            "graph": [
                {"name": "graph1"},
                {"name": "graph2"},
                {"name": "graph3"}
            ]
        }
        test_json = json.dumps(valid_json_content)
        with patch("msit_opcheck.graph_parser.ms_open", mock_open(read_data=test_json)) as mock_file:
            result = list(get_ge_graph_name("dummy_path"))
            self.assertEqual(result, ["graph1", "graph2", "graph3"])
            mock_file.assert_called_once_with("dummy_path", max_size=209715200)


class TestGetAllOpInfo(unittest.TestCase):
    def test_happy_path(self):
        """Test normal case where graph exists and has ops"""
        test_data = {
            "graph": [{
                "name": "test_graph",
                "op": [{"name": "op1"}, {"name": "op2"}],
                "attr": "graph_attr"
            }],
            "attr": "global_attr"
        }
        with patch("msit_opcheck.graph_parser.ms_open", mock_open(read_data=json.dumps(test_data))):
            result = get_all_opinfo("test.json", "test_graph")
        self.assertIsNotNone(result)


class TestOpInfo(unittest.TestCase):
    def test_opinfo_init(self):
        op_info_dict = {
            "input_desc": [{"name": "input1", "type": "float"}, {"name": "input2", "type": "int"}],
            "output_desc": []
        }
        op_info_instance = OpInfo(op_info_dict)
        self.assertEqual(len(op_info_instance.input_desc_list), 2)
        
    def test_update_op_type(self):
        op_info_dict = {
            "attr": [{"key": "_datadump_original_op_types", "value": {"list": {"s": ["Conv2D", "Relu"]}}}],
            "type": "Add"
        }
        op_info_instance = OpInfo(op_info_dict)
        op_info_instance.update_op_type()
        self.assertEqual(op_info_instance.op_type, ["Conv2D", "Relu"])