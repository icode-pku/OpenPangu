import unittest
from unittest.mock import patch, MagicMock
import sys


class TestTensorFlowGraphBuilder(unittest.TestCase):

    def setUp(self):
        self.backup_modules = {}
        self.modules_to_mock = [
            'tensorflow',
            'tensorflow.compat',
            'tensorflow.compat.v1',
            'autofuse',
            'autofuse.pyautofuse'
        ]
        for mod in self.modules_to_mock:
            if mod in sys.modules:
                self.backup_modules[mod] = sys.modules[mod]
            sys.modules[mod] = MagicMock()

        from msit_opcheck.autofuse.tf_builder import TensorFlowGraphBuilder, sanitize_filename, convert_to_tf_graph
        self.TensorFlowGraphBuilder, self.sanitize_filename = TensorFlowGraphBuilder, sanitize_filename
        self.convert_to_tf_graph = convert_to_tf_graph
        
        self.sample_description = """
                            Graph: test_graph
                            z0(10) : 16
                            z1(20) : 32
                            node1: Data (1)
                            .y.dtype = float32
                            .y.repeats = {1, 2, 3}
                            node2: Load (2)
                            .x = node1
                            node3: Output (3)
                            .x = node2
                            """

    def tearDown(self):
        for mod in self.modules_to_mock:
            if mod in self.backup_modules:
                sys.modules[mod] = self.backup_modules[mod]
            else:
                del sys.modules[mod]  # 删除新增的mock模块
        # 清理测试中导入的模块
        for mod in list(sys.modules.keys()):
            if mod.startswith("msit_opcheck.autofuse.tf_builder"):
                del sys.modules[mod]

    def test_sanitize_filename_given_normal_name_when_sanitized_then_returns_valid(self):
        self.assertEqual(self.sanitize_filename("a/b/c"), "a_b_c")
        
    def test__compute_diff_axes_given_identical_shapes_when_computed_then_returns_empty(self):
        result = self.TensorFlowGraphBuilder._compute_diff_axes([1,2,3], [1,2,3])
        self.assertEqual(result, [])
    
    def test__compute_diff_axes_given_diff_shapes_when_computed_then_returns_diff_axes(self):
        result = self.TensorFlowGraphBuilder._compute_diff_axes([1,2], [3,4])
        self.assertEqual(result, [0,1])
        
    def test__compute_diff_axes_given_start_index_when_computed_then_applies_offset(self):
        result = self.TensorFlowGraphBuilder._compute_diff_axes([1,2], [3,4], 2)
        self.assertEqual(result, [2,3])
    
    def test_parse_axes_given_valid_input_when_parsed_then_sets_axes(self):
        builder = self.TensorFlowGraphBuilder(self.sample_description)
        self.assertEqual(builder.axes, {'z0': 16, 'z1': 32})
        
    def test_parse_axes_given_invalid_line_when_parsed_then_ignores_line(self):
        desc = "invalid_line\n" + self.sample_description
        builder = self.TensorFlowGraphBuilder(desc)
        self.assertEqual(builder.axes, {'z0': 16, 'z1': 32})
    
    def test_parse_nodes_given_long_line_when_parsed_then_logs_warning(self):
        long_line = "a" * 3000 + "\n"
        with patch('msit_opcheck.autofuse.tf_builder.logger') as mock_logger:
            self.TensorFlowGraphBuilder(long_line + self.sample_description)
            mock_logger.warning.assert_called()
    
    def test_parse_nodes_given_graph_line_when_parsed_then_sets_graph_name(self):
        builder = self.TensorFlowGraphBuilder(self.sample_description)
        self.assertEqual(builder.graph_name, "test_graph")
        
    def test_parse_nodes_given_data_node_when_parsed_then_sets_attributes(self):
        builder = self.TensorFlowGraphBuilder(self.sample_description)
        node = builder.nodes['node1']
        self.assertEqual(node['attributes'], {'dtype': 'float32', 'repeats': [1,2,3]})
    
    def test_parse_nodes_given_multi_input_node_when_parsed_then_sets_inputs(self):
        desc = self.sample_description + """
                            node4: Add (4)
                            .x1 = node1
                            .x2 = node2
                            """
        builder = self.TensorFlowGraphBuilder(desc)
        self.assertEqual(builder.nodes['node4']['inputs'], ['node1','node2'])
    
    def test__build_data_node_given_valid_input_when_built_then_creates_placeholder(self):
        node_info = {
            "name": "test_data", 
            "attributes": {"dtype": "float32", "repeats": [1,2]}
        }
        with patch('tensorflow.compat.v1.placeholder') as mock_placehold:
            result = self.TensorFlowGraphBuilder._build_data_node(node_info)
            mock_placehold.assert_called_with(dtype="float32", shape=[1,2], name="test_data")
        
    def test_build_tensorflow_nodes_given_unsupported_node_when_built_then_raises_error(self):
        desc = self.sample_description + "nodeX: UnknownType (99)\n"
        with self.assertRaises(ValueError):
            self.TensorFlowGraphBuilder(desc)
    
    def test__build_output_node_given_valid_input_when_built_then_adds_to_outputs(self):
        builder = self.TensorFlowGraphBuilder(self.sample_description)
        builder.nodes['node3']['output'] = MagicMock()
        self.assertEqual(len(builder.output_nodes), 1)
        self.assertEqual(builder.output_nodes[0], 'node3')
    
    def test__build_sum_mode_given_diff_shapes_when_built_then_computes_reduce_sum(self):
        node_info = {
            "name": "test_sum",
            "inputs": ["node1"],
            "attributes": {"repeats": [1]}
        }
        mock_input = MagicMock()
        mock_input.shape.as_list.return_value = [1,2,3]
        builder = self.TensorFlowGraphBuilder(self.sample_description)
        builder.nodes = {"node1": {"output": mock_input}}
        
        with patch.object(self.TensorFlowGraphBuilder, '_compute_diff_axes', return_value=[1,2]) as mock_diff, \
             patch("tensorflow.compat.v1.reduce_sum") as mock_reduce_sum:
            result = builder._build_sum_mode(node_info)
            mock_diff.assert_called_once()
            mock_reduce_sum.assert_called_with(mock_input, name="test_sum", axis=[1,2])
    
    def test_get_nodes_given_built_nodes_when_called_then_returns_outputs(self):
        builder = self.TensorFlowGraphBuilder(self.sample_description)
        builder.nodes = {'n1': {'output': 't1'}, 'n2': {'output': 't2'}}
        self.assertEqual(builder.get_nodes(), {'n1':'t1','n2':'t2'})
    
    def test_get_output_nodes_given_outputs_when_called_then_returns_outputs(self):
        builder = self.TensorFlowGraphBuilder(self.sample_description)
        builder.output_nodes = ['n1']
        builder.nodes = {'n1': {'output': 't1'}}
        self.assertEqual(builder.get_output_nodes(), ['t1'])
    
    def test_list_placeholders_given_no_placeholders_when_called_then_returns_empty(self):
        mock_tf = sys.modules['tensorflow.compat.v1']
        mock_graph = MagicMock()
        mock_tf.Graph.return_value = mock_graph
        mock_graph.as_graph_def.return_value.node = [MagicMock(op='Const')]
        builder = self.TensorFlowGraphBuilder(self.sample_description)
        self.assertEqual(builder.list_placeholders(), [])
    
    def test_convert_to_tf_graph_given_valid_input_when_called_then_returns_builder(self):
        mock_pyautofuse = MagicMock()
        mock_pyautofuse.ascir.utils.debug_str.return_value = "graph_desc"
        mock_pyautofuse = MagicMock()
        mock_pyautofuse.graph = MagicMock()
        with patch.dict(sys.modules, {'autofuse.pyautofuse': mock_pyautofuse}):
            from msit_opcheck.autofuse.tf_builder import convert_to_tf_graph
            builder = convert_to_tf_graph(mock_pyautofuse)
            self.assertIsInstance(builder, self.TensorFlowGraphBuilder)
