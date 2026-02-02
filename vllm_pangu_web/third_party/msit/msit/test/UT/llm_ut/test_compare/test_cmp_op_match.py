import pytest
import unittest
from unittest.mock import MagicMock, patch

import msit_llm.dump.torch_dump.topo as topo
from msit_llm.compare.cmp_op_match import OpMatchMap, MatchLocation
from msit_llm.dump.torch_dump.topo import TreeNode
from msit_llm.compare.cmp_op_match import policy_enhanced_name_match, policy_layer_type_cnt_match, \
                                            policy_name_full_match, policy_output, \
                                            policy_layer_special_match, policy_rope_operator_match, \
                                            policy_module_match, OpMatchMgr, policy_qwen_match
from msit_llm.dump.torch_dump.topo import TreeNode
from msit_llm.compare.op_mapping import QWEN_OP_MAPPING


@pytest.fixture
def patch_min_layer_number():
    # 记录原始值
    original_value = topo.MIN_LAYER_NUMBER
    # 修改为测试值
    topo.MIN_LAYER_NUMBER = 0
    yield
    # 恢复原始值
    topo.MIN_LAYER_NUMBER = original_value


def test_policy_name_full_match():

    golden_root = TreeNode('root.lm_head', 'root.lm_head')
    golden_child1 = TreeNode('child1', 'golden_operation1', show_order=1)
    golden_child2 = TreeNode('child2', 'golden_operation2', show_order=2)
    golden_root.add_child(golden_child1)
    golden_root.add_child(golden_child2)

    my_root = TreeNode('LmHead', 'LmHead')
    my_child1 = TreeNode('child1', 'my_operation1', show_order=1)
    my_child2 = TreeNode('child2', 'my_operation2', show_order=2)
    my_root.add_child(my_child1)
    my_root.add_child(my_child2)

    match_map = OpMatchMap(golden_root, my_root)
    policy_name_full_match(golden_root, my_root, match_map)
    matches = match_map.get_match_map(enable_print=True)

    expected_matches = [
        (my_child1, MatchLocation.ALL_OUTPUT, golden_child1, MatchLocation.ALL_OUTPUT),
        (my_child2, MatchLocation.ALL_OUTPUT, golden_child2, MatchLocation.ALL_OUTPUT)
    ]
    assert len(matches) == len(expected_matches)
    for match, expected in zip(matches, expected_matches):
        assert match == expected


def test_policy_output():

    golden_root = TreeNode('lm_head', 'root.lm_head')
    golden_child1 = TreeNode('child1', 'golden_operation1', show_order=1)
    golden_child2 = TreeNode('child2', 'golden_operation2', show_order=2)
    golden_grandchild1 = TreeNode('grandchild1', 'golden_operation3', show_order=3)
    golden_root.add_child(golden_child1)
    golden_root.add_child(golden_child2)
    golden_child2.add_child(golden_grandchild1)

    my_root = TreeNode('LmHead', 'Lmhead')
    my_child1 = TreeNode('child1', 'my_operation1', show_order=1)
    my_child2 = TreeNode('child2', 'my_operation2', show_order=2)
    my_grandchild1 = TreeNode('grandchild1', 'my_operation3', show_order=3)
    my_root.add_child(my_child1)
    my_root.add_child(my_child2)
    my_child2.add_child(my_grandchild1)

    match_map = OpMatchMap(golden_root, my_root)
    policy_output(golden_root, my_root, match_map)
    matches = match_map.get_match_map(enable_print=True)

    expected_matches = [
        (my_child2, MatchLocation.ALL_OUTPUT, golden_child2, MatchLocation.ALL_OUTPUT),
        (my_child2, MatchLocation.ALL_OUTPUT, golden_child2.children[0], MatchLocation.ALL_OUTPUT),
        (my_child2.children[0], MatchLocation.ALL_OUTPUT, golden_child2, MatchLocation.ALL_OUTPUT),
        (my_child2.children[0], MatchLocation.ALL_OUTPUT, golden_child2.children[0], MatchLocation.ALL_OUTPUT)
    ]

    assert len(matches) == len(expected_matches)
    for match, expected in zip(matches, expected_matches):
        assert match == expected


def test_policy_enhanced_name_match(patch_min_layer_number):

    assert topo.MIN_LAYER_NUMBER == 0

    golden_root = TreeNode('lm_head', 'root.lm_head')
    golden_mlp = TreeNode('golden_mlp', 'mlp', tensor_path='my_golden_path')
    golden_rmsnorm = TreeNode('golden_rmsnorm', 'rmsnormoperation', 
                              tensor_path='torch_dump/root.mlp.post_attention_layernorm')
    golden_split = TreeNode('my_act', 'activationoperation', tensor_path='torch_dump/root.mlp.act_fn')
    golden_mlp.add_child(golden_rmsnorm)
    golden_mlp.add_child(golden_split)
    golden_root.add_child(golden_mlp)

    my_root = TreeNode('LmHead', 'LmHead')
    my_mlp = TreeNode('my_mlp', 'mlp', tensor_path='my_mlp_path')
    my_rmsnorm = TreeNode('my_rmsnorm', 'rmsnormoperation', 
                          tensor_path='2_MlpGateUpWeightPack/0_NormLinear/0_RmsNormOperation')
    my_split = TreeNode('my_act', 'activationoperation', tensor_path='2_MlpGateUpWeightPack/2_ActivationOperation')
    my_mlp.add_child(my_rmsnorm)
    my_mlp.add_child(my_split) 
    my_root.add_child(my_mlp)

    match_map = OpMatchMap(golden_root, my_root)
    policy_enhanced_name_match(golden_root, my_root, match_map)
    matches = match_map.get_match_map(enable_print=True)

    expected_matches = [
        (my_mlp.children[0], MatchLocation.ALL_OUTPUT, golden_mlp.children[0], MatchLocation.ALL_OUTPUT),
        (my_mlp.children[1], MatchLocation.ALL_OUTPUT, golden_mlp.children[1], MatchLocation.ALL_OUTPUT)
    ]

    assert len(matches) == len(expected_matches)
    for match, expected in zip(matches, expected_matches):
        assert match == expected


def test_policy_layer_type_cnt_match():

    golden_root = TreeNode('lm_head', 'root.lm_head')
    golden_child1 = TreeNode('child1', 'LinearOperation', show_order=1)
    golden_child2 = TreeNode('child2', 'LinearNoQuant', show_order=2)
    golden_root.add_child(golden_child1)
    golden_root.add_child(golden_child2)

    my_root = TreeNode('LmHead', 'LmHead')
    my_child1 = TreeNode('child1', 'LinearQuantOperation', show_order=1)
    my_child2 = TreeNode('child2', 'LinearDequantOnly', show_order=2)
    my_root.add_child(my_child1)
    my_root.add_child(my_child2)

    match_map = OpMatchMap(golden_root, my_root)
    policy_layer_type_cnt_match(golden_root, my_root, match_map)
    matches = match_map.get_match_map(enable_print=True)

    expected_matches = [
        (my_root, MatchLocation.ALL_OUTPUT, golden_root, MatchLocation.ALL_OUTPUT),
        (my_child1, MatchLocation.ALL_OUTPUT, golden_child1, MatchLocation.ALL_OUTPUT),
        (my_child2, MatchLocation.ALL_OUTPUT, golden_child2, MatchLocation.ALL_OUTPUT)
    ]

    assert len(matches) == len(expected_matches)
    for match, expected in zip(matches, expected_matches):
        assert match == expected




def test_policy_layer_special_match():

    golden_root = TreeNode('model', 'Baichuan')
    golden_child1 = TreeNode('embed_tokens', 'golden_operation1')
    golden_child2 = TreeNode('root.model.norm', 'golden_operation2')
    golden_child3 = TreeNode('lm_head', 'golden_operation3')
    golden_root.add_child(golden_child1)
    golden_root.add_child(golden_child2)
    golden_root.add_child(golden_child3)
    
    my_root = TreeNode('model', 'BaichuanModel')
    my_child1 = TreeNode('wordembedding', 'my_operation1')
    my_child2 = TreeNode('RmsNormOperation_123', 'my_operation2')
    my_child3 = TreeNode('lmhead', 'my_operation3')
    my_root.add_child(my_child1)
    my_root.add_child(my_child2)
    my_root.add_child(my_child3)

    match_map = OpMatchMap(golden_root, my_root)
    policy_layer_special_match(golden_root, my_root, match_map)
    matches = match_map.get_match_map(enable_print=True)

    expected_matches = [
        (my_child1, MatchLocation.ALL_OUTPUT, golden_child1, MatchLocation.ALL_OUTPUT),
        (my_child2, MatchLocation.ALL_OUTPUT, golden_child2, MatchLocation.ALL_OUTPUT),
        (my_child3, MatchLocation.ALL_OUTPUT, golden_child3, MatchLocation.ALL_OUTPUT)
    ]

    assert len(matches) == len(expected_matches)
    for match, expected in zip(matches, expected_matches):
        assert match == expected


def test_policy_rope_operator_match(patch_min_layer_number):

    golden_root = TreeNode('model', 'Baichuan')
    golden_child1 = TreeNode('rotary1', 'rotary')
    golden_child2 = TreeNode('rotary2', 'rotary')
    golden_grandchild = TreeNode('rotary3', 'rotary')
    golden_root.add_child(golden_child1)
    golden_root.add_child(golden_child2)
    golden_child2.add_child(golden_grandchild)
    
    my_root = TreeNode('model', 'BaichuanModel')
    my_child1 = TreeNode('ropeoperation1', '1_RotaryPositionEmbedding')
    my_child2 = TreeNode('ropeoperation2', '1_RotaryPositionEmbedding')
    my_grandchild = TreeNode('ropeoperation3', '1_RotaryPositionEmbedding')
    my_root.add_child(my_child1)
    my_root.add_child(my_child2)
    my_child2.add_child(my_grandchild)

    match_map = OpMatchMap(golden_root, my_root)
    policy_rope_operator_match(golden_root, my_root, match_map)
    matches = match_map.get_match_map(enable_print=True)

    expected_matches = [
        (my_grandchild, MatchLocation.ALL_INPUT, golden_grandchild, MatchLocation.ALL_OUTPUT)
    ]

    assert len(matches) == len(expected_matches)
    for match, expected in zip(matches, expected_matches):
        assert match == expected  


def test_policy_module_match(patch_min_layer_number):
    assert topo.MIN_LAYER_NUMBER == 0
    # 创建测试用的树
    golden_root = TreeNode('model', 'Baichuan')
    golden_child1 = TreeNode('child1', 'decoder', tensor_path='transformer/self_attn')
    golden_grandchild1 = TreeNode('grandchild1', 'self_attn', tensor_path='transformer/layer2/self_attn')
    golden_grandchild2 = TreeNode('grandchild2', 'mlp', tensor_path='transformer/layer2/mlp')
    
    golden_child1.add_child(golden_grandchild1)
    golden_child1.add_child(golden_grandchild2)
    golden_root.add_child(golden_child1)

    my_root = TreeNode('model', 'BaichuanModel')
    my_child1 = TreeNode('child1', 'root', tensor_path='transformer/Attention')
    my_grandchild1 = TreeNode('grandchild1', 'Attention', tensor_path='transformer/layer2/Attention')
    my_grandchild2 = TreeNode('grandchild2', 'Mlp', tensor_path='transformer/layer2/Mlp')
    
    my_child1.add_child(my_grandchild1)
    my_child1.add_child(my_grandchild2)
    my_root.add_child(my_child1)
    # 执行匹配策略
    match_map = OpMatchMap(golden_root, my_root)
    policy_module_match(golden_root, my_root, match_map)
    matches = match_map.get_match_map(enable_print=True)
    expected_matches = [
        (my_grandchild1, MatchLocation.ALL_OUTPUT, golden_grandchild1, MatchLocation.ALL_OUTPUT),
        (my_grandchild2, MatchLocation.ALL_OUTPUT, golden_grandchild2, MatchLocation.ALL_OUTPUT),
    ]
    assert len(matches) == len(expected_matches)
    for match, expected in zip(matches, expected_matches):
        assert match == expected


class TestOpMatchMgrInit(unittest.TestCase):
    def setUp(self):
        # Common mock args setup
        self.args = MagicMock()
        self.args.cmp_level = None
        self.args.stats = False

    def test_init_with_cmp_level_layer(self):
        """Test initialization with cmp_level='layer'"""
        self.args.cmp_level = "layer"
        mgr = OpMatchMgr(self.args)
        
        self.assertEqual(len(mgr.selected_policies), 1)
        self.assertEqual(mgr.selected_policies[0], mgr.op_match_policies_layer)
        self.assertFalse(mgr.cmp_all)
    
    def test_init_with_cmp_level_module(self):
        """Test initialization with cmp_level='module'"""
        self.args.cmp_level = "module"
        mgr = OpMatchMgr(self.args)
        
        self.assertEqual(len(mgr.selected_policies), 1)
        self.assertEqual(mgr.selected_policies[0], mgr.op_match_policies_module)
        self.assertFalse(mgr.cmp_all)

    def test_init_with_cmp_level_api(self):
        """Test initialization with cmp_level='api'"""
        self.args.cmp_level = "api"
        mgr = OpMatchMgr(self.args)
        
        self.assertEqual(len(mgr.selected_policies), 1)
        self.assertEqual(mgr.selected_policies[0], mgr.op_match_policies)
        self.assertFalse(mgr.cmp_all)

    def test_init_with_cmp_level_logits(self):
        """Test initialization with cmp_level='logits'"""
        self.args.cmp_level = "logits"
        mgr = OpMatchMgr(self.args)
        
        self.assertEqual(len(mgr.selected_policies), 1)
        self.assertEqual(mgr.selected_policies[0], mgr.op_match_policies)
        self.assertFalse(mgr.cmp_all)

    def test_init_with_stats_true(self):
        """Test initialization with stats=True"""
        self.args.stats = True
        mgr = OpMatchMgr(self.args)
        
        self.assertEqual(len(mgr.selected_policies), 1)
        self.assertEqual(mgr.selected_policies[0], mgr.op_match_policies)
        self.assertFalse(mgr.cmp_all)

    def test_init_with_no_special_args(self):
        """Test initialization with no special args (cmp_all case)"""
        mgr = OpMatchMgr(self.args)
        
        self.assertEqual(len(mgr.selected_policies), 3)
        self.assertEqual(mgr.selected_policies[0], mgr.op_match_policies_layer)
        self.assertEqual(mgr.selected_policies[1], mgr.op_match_policies_module)
        self.assertEqual(mgr.selected_policies[2], mgr.op_match_policies)
        self.assertTrue(mgr.cmp_all)

    def test_policy_lists_initialized(self):
        """Test that all policy lists are properly initialized"""
        mgr = OpMatchMgr(self.args)
        
        self.assertTrue(hasattr(mgr, 'op_match_policies'))
        self.assertTrue(hasattr(mgr, 'op_match_policies_layer'))
        self.assertTrue(hasattr(mgr, 'op_match_policies_module'))
        self.assertIsInstance(mgr.op_match_policies, list)
        self.assertIsInstance(mgr.op_match_policies_layer, list)
        self.assertIsInstance(mgr.op_match_policies_module, list)


class TestPolicyQwenMatch(unittest.TestCase):
    def setUp(self):
        # Create mock TreeNode objects
        self.mock_golden_root = MagicMock(spec=TreeNode)
        self.mock_my_root = MagicMock(spec=TreeNode)
        self.mock_match_map = MagicMock()
        
        # Setup default return values
        self.mock_golden_root.get_all_nodes.return_value = []
        self.mock_my_root.get_all_nodes.return_value = []
        self.mock_golden_root.get_layer_node_type.return_value = "golden_layer"
        self.mock_my_root.get_layer_node_type.return_value = "my_layer"
        self.mock_golden_root.get_layer_node.return_value = []
        self.mock_my_root.get_layer_node.return_value = []

    def test_empty_inputs(self):
        """Test with empty input trees"""
        policy_qwen_match(self.mock_golden_root, self.mock_my_root, self.mock_match_map)
        self.mock_match_map.add_score.assert_not_called()

    def test_basic_matching(self):
        """Test basic matching scenario with one matching pair"""
        # Setup test data
        golden_node = MagicMock()
        golden_node.tensor_path = "golden/path/to/op"
        my_node = MagicMock()
        my_node.tensor_path = "my/path/to/qkv"
        
        # Mock layer nodes
        golden_layer_node = MagicMock()
        golden_layer_node.get_next_level_nodes.return_value = [golden_node]
        my_layer_node = MagicMock()
        my_layer_node.get_next_level_nodes.return_value = [my_node]
        
        # Configure mocks
        self.mock_golden_root.get_all_nodes.return_value = [golden_node]
        self.mock_my_root.get_all_nodes.return_value = [my_node]
        self.mock_golden_root.get_layer_node.return_value = [golden_layer_node]
        self.mock_my_root.get_layer_node.return_value = [my_layer_node]
        
        # Setup QWEN_OP_MAPPING for test
        with patch.dict('msit_llm.compare.op_mapping.QWEN_OP_MAPPING', 
                       {'qkv': ['op']}, clear=True):
            policy_qwen_match(self.mock_golden_root, self.mock_my_root, self.mock_match_map)
        
        # Verify match was called
        self.mock_match_map.add_score.assert_called_once()


    def test_multiple_matches(self):
        """Test scenario with multiple matching pairs"""
        # Create test nodes
        golden_nodes = [MagicMock() for _ in range(3)]
        for i, node in enumerate(golden_nodes):
            node.tensor_path = f"golden/path/op_{i}"
        
        my_nodes = [MagicMock() for _ in range(3)]
        for i, node in enumerate(my_nodes):
            node.tensor_path = f"my/path/qkv_{i}"
        
        # Mock layer nodes
        golden_layer_node = MagicMock()
        golden_layer_node.get_next_level_nodes.return_value = golden_nodes
        my_layer_node = MagicMock()
        my_layer_node.get_next_level_nodes.return_value = my_nodes
        
        # Configure mocks
        self.mock_golden_root.get_all_nodes.return_value = golden_nodes
        self.mock_my_root.get_all_nodes.return_value = my_nodes
        self.mock_golden_root.get_layer_node.return_value = [golden_layer_node]
        self.mock_my_root.get_layer_node.return_value = [my_layer_node]
        
        # Setup QWEN_OP_MAPPING for test
        with patch.dict('msit_llm.compare.op_mapping.QWEN_OP_MAPPING', 
                       {'qkv': ['op']}, clear=True):
            policy_qwen_match(self.mock_golden_root, self.mock_my_root, self.mock_match_map)
        
        # Verify matches were called
        self.assertEqual(self.mock_match_map.add_score.call_count, 9)
