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

import tempfile
from unittest import TestCase
import torch.nn as nn

from msit_llm.transform.model_parser.parser import (
    parse_input_names,
    fix_parsed_model,
    build_model_tree,
    get_weight_names,
    parse_input_max_count,
    parse_by_idx,
    get_atb_model_names,
    process_layer,
)

from msit_llm.transform.model_parser.parser import parse_input_names


class SimpleModel_1(nn.Module):
    def __init__(self):
        super(SimpleModel_1, self).__init__()
        self.linear = nn.Linear(10, 20)
        self.dropout = nn.Dropout(0.5)


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 30))


class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.attention = nn.MultiheadAttention(10, 1)


class SimpleModel_2(nn.Module):
    def __init__(self, with_rope):
        super(SimpleModel_2, self).__init__()
        self.embed = nn.Embedding(10, 10)
        self.linear = nn.Linear(10, 10)
        self.dropout = nn.Dropout(0.5)
        self.config = type("Config", (object,), {"model_type": "qwen_model"})


class SimpleModel_3(nn.Module):
    def __init__(self):
        super(SimpleModel_3, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Sequential(nn.Linear(20, 30), nn.Dropout(), nn.Linear(30, 40))
        self.attention = nn.MultiheadAttention(60, 6)
        self.norm1 = nn.LayerNorm(60)
        self.norm2 = nn.LayerNorm(60)


class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


class NoConfigModule(nn.Module):
    def __init__(self):
        super(NoConfigModule, self).__init__()
        self.linear = nn.Linear(10, 20)

    def forward(self, x):
        return self.linear(x)


LLAMA_CASE_01 = """
DecoderModel::~DecoderModel() {}

std::map<std::string, std::vector<std::string>> GetLlamaModelInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> llamaInTensorCandiadates = {
        {"default", {
            "input_ids", "input_embedding", "positional_ids", "cosine_table", "sine_table", "attention_mask",
            "block_tables", "slots", "kv_cache_idx", "token_offset", "place_holder", "seq_len", "logits_indices"}
        },
        {"compress_head", {"batch_wins", "ra_seq_len"}},
        {"speculate", {"q_len"}},
        {"lora_common", {"seq_len_cum_sum"}},
        {"lora_per_layer", {
            "qkv_lora_a_0", "qkv_lora_b_0", "qkv_lora_a_1", "qkv_lora_b_1",
            "qkv_lora_a_2", "qkv_lora_b_2", "qkv_dense_lora_a", "qkv_dense_lora_b",
            "mlp_lora_a_0", "mlp_lora_b_0", "mlp_lora_a_1", "mlp_lora_b_1",
            "mlp_down_lora_a", "mlp_down_lora_b"}
        },
    };
    return llamaInTensorCandiadates;
}
"""

LLAMA_CASE_02 = """
enum InTensorId : int {
    // define inTensor
    // idx: 0, input_ids, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_INPUT = 0,
    // idx: 1, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_POSITION_IDS,

    // idx: 2, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_COS_TABLE,
    // idx: 3, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_SIN_TABLE,
    // idx: 4, shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings]
    // PA: [maxInputLength, maxInputLength]
    IN_TENSOR_ATTENTION_MASK,
    // idx: 5, shape: [4, 9]; PA所需入参
    IN_TENSOR_BLOCK_TABLES,
    // idx: 6, shape: [seqLen]; PA所需入参
    IN_TENSOR_SLOTS,
    // idx: 7, shape: [1]; FA所需入参
    IN_TENSOR_KV_CACHE_IDX,
    // idx: 8, shape: [batchSize]; FA所需入参
    IN_TENSOR_TOKEN_OFFSET,
    // idx: 9, shape: FA: [batchSize] PA: [4]
    IN_TENSOR_SEQ_LENGTHS,
    // idx: 10, shape: FA: [batchSize]  PA: [4]
    IN_TENSOR_LOGTIS_INDICES,

    IN_PLACEHOLDER,
    IN_TENSOR_Q_LEN,
    IN_TENSOR_MAX,
};
"""

QWEN_CASE_01 = """
enum InTensorId : int {
    IN_TENSOR_INPUT = 0,  // input_ids or word_embedding
    IN_TENSOR_POSITIONIDS,
    IN_TENSOR_COS,  // cos_table or cos_embed
    IN_TENSOR_SIN,  // sin_table or sin_embed
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_BLOCK_TABLES,
    IN_TENSOR_SLOTS,
    IN_TENSOR_KV_CACHE_IDX,
    IN_TENSOR_TOKEN_OFFSET,
    IN_TENSOR_SEQ_LENGTHS,
    IN_TENSOR_LOGTIS_INDICES,
    IN_PLACEHOLDER,
    IN_TENSOR_Q_LEN,
    IN_TENSOR_MAX,
};

enum InternalTensorId : int {
    INTERNAL_HIDDENSTATES = 0,
    INTERNAL_COSEMBED,
    INTERNAL_SINEMBED,
    INTERNAL_TENSOR_MAX,
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};
"""

BLOOM_CASE_01 = """

enum DecoderModelTensorIdx : uint32_t {
    // define inTensor
    // idx: 0, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_INPUT_IDS = 0,
    // idx: 1, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_POSITION_IDS,
    // idx: 2, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_COS_TABLE,
    // idx: 3, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_SIN_TABLE,
    // idx: 4, shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings]
    // PA: [maxInputLength, maxInputLength]
    IN_TENSOR_ATTENTION_MASK,
    // idx: 5, shape: [4, 9]; PA所需入参
    IN_TENSOR_BLOCK_TABLES,
    // idx: 6, shape: [seqLen]; PA所需入参
    IN_TENSOR_SLOTS,
    // idx: 7, shape: [1]; FA所需入参
    IN_TENSOR_LAYER_IDX,
    // idx: 8, shape: [batchSize]; FA所需入参
    IN_TENSOR_TOKEN_OFFSET,
    // idx: 9, shape: [1]
    IN_TENSOR_PLACE_HOLDER,
    // idx: 10, shape: FA: [batchSize] PA: [4]
    IN_TENSOR_SEQ_LEN,
    // idx: 11, shape: FA: [batchSize]  PA: [4]
    IN_TENSOR_LOGTIS_INDICES,
};

enum DecoderModelInternalTensorIdx : uint32_t {
    INTERNAL_TENSOR_HIDDEN_STATES = 0,
    // INTERNAL_TENSOR_TO_LAYERS
};
"""

CHATGLM_CASE_01 = """

enum DecoderModelTensorIdx : uint32_t {
    // define inTensor
    // idx: 0, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_INPUT_IDS = 0,
    // idx: 1, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_POSITION_IDS,
    // idx: 2, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_COS_TABLE,
    // idx: 3, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_SIN_TABLE,
    // idx: 4, shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings]
    // PA: [maxInputLength, maxInputLength]
    IN_TENSOR_ATTENTION_MASK,
    // idx: 5, shape: [4, 9]; PA所需入参
    IN_TENSOR_BLOCK_TABLES,
    // idx: 6, shape: [seqLen]; PA所需入参
    IN_TENSOR_SLOTS,
    // idx: 7, shape: [1]; FA所需入参
    IN_TENSOR_KV_CACHE_IDX,
    // idx: 8, shape: [batchSize]; FA所需入参
    IN_TENSOR_TOKEN_OFFSET,
    // idx: 9, shape: [1]
    IN_TENSOR_PLACE_HOLDER,
    // idx: 10, shape: FA: [batchSize] PA: [4]
    IN_TENSOR_SEQ_LEN,
    // idx: 11, shape: FA: [batchSize]  PA: [4]
    IN_TENSOR_LOGTIS_INDICES,
    // idx 12, shape: [batchsize] dtype: int32
    IN_TENSOR_Q_LEN,
};

enum DecoderModelInternalTensorIdx : uint32_t {
    // define internelTensor
    // idx: 0, shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
    INTERNAL_TENSOR_HIDDEN_STATES = 0,
    // idx: 1, shape: [batchSize * seqLen, hiddenSizePerAttentionHead]
    INTERNAL_TENSOR_COS_EMB,
    // idx: 2, shape: [batchSize * seqLen, hiddenSizePerAttentionHead]
    INTERNAL_TENSOR_SIN_EMB,
};
"""

MODEL_CASE_01 = """

enum DecoderModelTensorIdx : uint32_t {
    // define inTensor
    // idx: 0, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_INPUT_IDS = 0,
    // idx: 1, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_POSITION_IDS,
    // idx: 2, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_COS_TABLE,
    // idx: 3, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_SIN_TABLE,
    // idx: 4, shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings]
    // PA: [maxInputLength, maxInputLength]
    IN_TENSOR_ATTENTION_MASK,
    // idx: 5, shape: [4, 9]; PA所需入参
    IN_TENSOR_BLOCK_TABLES,
    // idx: 6, shape: [seqLen]; PA所需入参
    IN_TENSOR_SLOTS,
    // idx: 7, shape: [1]; FA所需入参
    IN_TENSOR_LAYER_IDX,
    // idx: 8, shape: [batchSize]; FA所需入参
    IN_TENSOR_TOKEN_OFFSET,
    // idx: 9, shape: [1]
    IN_TENSOR_PLACE_HOLDER,
    // idx: 10, shape: FA: [batchSize] PA: [4]
};

enum DecoderModelInternalTensorIdx : uint32_t {
    // define internelTensor
    // idx: 0, shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
    INTERNAL_TENSOR_HIDDEN_STATES = 0,
    // idx: 1, shape: [batchSize * seqLen, hiddenSizePerAttentionHead]
    INTERNAL_TENSOR_COS_EMB,
};
"""


class TestParserReg(TestCase):
    def test_llama(self):
        self.assertEqual(
            [
                "input_ids",
                "input_embedding",
                "positional_ids",
                "cosine_table",
                "sine_table",
                "attention_mask",
                "block_tables",
                "slots",
                "kv_cache_idx",
                "token_offset",
                "place_holder",
                "seq_len",
                "logits_indices",
            ],
            parse_input_names(LLAMA_CASE_01),
        )

    def test_bloom(self):
        self.assertEqual(
            [
                "IN_TENSOR_INPUT_IDS",
                "IN_TENSOR_POSITION_IDS",
                "IN_TENSOR_COS_TABLE",
                "IN_TENSOR_SIN_TABLE",
                "IN_TENSOR_ATTENTION_MASK",
                "IN_TENSOR_BLOCK_TABLES",
                "IN_TENSOR_SLOTS",
                "IN_TENSOR_LAYER_IDX",
                "IN_TENSOR_TOKEN_OFFSET",
                "IN_TENSOR_PLACE_HOLDER",
                "IN_TENSOR_SEQ_LEN",
                "IN_TENSOR_LOGTIS_INDICES",
            ],
            parse_input_names(BLOOM_CASE_01),
        )

    def test_chatglm(self):
        self.assertEqual(
            [
                "IN_TENSOR_INPUT_IDS",
                "IN_TENSOR_POSITION_IDS",
                "IN_TENSOR_COS_TABLE",
                "IN_TENSOR_SIN_TABLE",
                "IN_TENSOR_ATTENTION_MASK",
                "IN_TENSOR_BLOCK_TABLES",
                "IN_TENSOR_SLOTS",
                "IN_TENSOR_KV_CACHE_IDX",
                "IN_TENSOR_TOKEN_OFFSET",
                "IN_TENSOR_PLACE_HOLDER",
                "IN_TENSOR_SEQ_LEN",
                "IN_TENSOR_LOGTIS_INDICES",
                "IN_TENSOR_Q_LEN",
            ],
            parse_input_names(CHATGLM_CASE_01),
        )

    def test_qwen(self):
        self.assertEqual(
            [
                "IN_TENSOR_INPUT",
                "IN_TENSOR_POSITIONIDS",
                "IN_TENSOR_COS",
                "IN_TENSOR_SIN",
                "IN_TENSOR_ATTENTIONMASK",
                "IN_TENSOR_BLOCK_TABLES",
                "IN_TENSOR_SLOTS",
                "IN_TENSOR_KV_CACHE_IDX",
                "IN_TENSOR_TOKEN_OFFSET",
                "IN_TENSOR_SEQ_LENGTHS",
                "IN_TENSOR_LOGTIS_INDICES",
                "IN_PLACEHOLDER",
            ],
            parse_input_names(QWEN_CASE_01),
        )


class TestFixParsedModel(TestCase):

    def test_fix_parsed_model_bloom(self):
        parsed_model = {"weight_names": {"model_name": "bloom", "word_embeddings": "embedding_weights"}}
        fix_parsed_model(parsed_model)
        expected_model = {
            "weight_names": {
                "model_name": "bloom",
                "word_embeddings": "embedding_weights",
                "lmhead": "embedding_weights",
            }
        }
        self.assertEqual(parsed_model, expected_model)

    def test_fix_parsed_model_qwen(self):
        parsed_model = {"weight_names": {"model_name": "qwen"}}
        fix_parsed_model(parsed_model)
        expected_model = {"weight_names": {"model_name": "qwen", "mlp_sep": ["w2", "w1"], "down_proj": "c_proj"}}
        self.assertEqual(parsed_model, expected_model)

    def test_fix_parsed_model_other(self):
        parsed_model = {"weight_names": {"model_name": "other"}}
        fix_parsed_model(parsed_model)
        expected_model = {"weight_names": {"model_name": "other"}}
        self.assertEqual(parsed_model, expected_model)

    def test_fix_parsed_model_no_model_name(self):
        parsed_model = {"weight_names": {}}
        fix_parsed_model(parsed_model)
        expected_model = {"weight_names": {}}
        self.assertEqual(parsed_model, expected_model)


class TestBuildModelTree(TestCase):
    def test_build_model_tree_simple(self):
        model = SimpleModel_1()
        tree = build_model_tree(model)
        expected_tree = {
            "name": "SimpleModel_1",
            "children": [{"name": "linear", "kind": "Linear", "in_features": 10, "out_features": 20, "bias": True}],
        }
        self.assertEqual(tree, expected_tree)

    def test_build_model_tree_with_mlp(self):
        model = MLPModel()
        tree = build_model_tree(model)
        expected_tree = {
            "name": "MLPModel",
            "children": [
                {"name": "0", "kind": "Linear", "in_features": 10, "out_features": 20, "bias": True},
                {"name": "ReLU", "children": []},
                {"name": "2", "kind": "Linear", "in_features": 20, "out_features": 30, "bias": True},
            ],
        }
        self.assertEqual(tree, expected_tree)

    def test_build_model_tree_with_attention(self):
        model = AttentionModel()
        tree = build_model_tree(model)
        expected_tree = {
            "name": "AttentionModel",
            "children": [{"name": "NonDynamicallyQuantizableLinear", "children": []}],
        }
        self.assertEqual(tree, expected_tree)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            build_model_tree("not a module")

    def test_empty_module(self):
        model = EmptyModule()
        expected_output = {"name": "EmptyModule", "children": [{"name": "EmptyModule", "children": []}]}
        self.assertEqual(build_model_tree(model), expected_output)


class TestGetWeightNames(TestCase):
    def test_get_weight_names_without_rope(self):
        model = self.create_model(with_rope=True)
        result = get_weight_names(model)
        self.assertIn("weight_names", result)
        weight_names = result["weight_names"]
        self.assertEqual(weight_names["pe_type"], "ALIBI")
        self.assertEqual(weight_names["model_name"], "qwen_model")
        self.assertEqual(weight_names["word_embeddings"], "embed")
        self.assertEqual(weight_names["lmhead"], "linear")

    def create_model(self, with_rope):
        return SimpleModel_2(with_rope)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            get_weight_names("not a module")

    def test_missing_config(self):
        model = NoConfigModule()
        with self.assertRaises(AttributeError):
            get_weight_names(model)


class TestParseInputMaxCount(TestCase):
    def test_normal_case_q_len_index(self):
        content = "IN_TENSOR_Q_LEN_INDEX = 42;"
        self.assertEqual(parse_input_max_count(content), 42)

    def test_normal_case_tensor_count(self):
        content = "IN_TENSOR_COUNT = 100;"
        self.assertEqual(parse_input_max_count(content), 100)

    def test_no_match(self):
        content = "No match here."
        self.assertEqual(parse_input_max_count(content), -1)

    def test_empty_string(self):
        content = ""
        self.assertEqual(parse_input_max_count(content), -1)

    def test_non_integer_value(self):
        content = "IN_TENSOR_Q_LEN_INDEX = abc;"
        self.assertEqual(parse_input_max_count(content), -1)

    def test_multiple_matches(self):
        content = "IN_TENSOR_Q_LEN_INDEX = 50; IN_TENSOR_COUNT = 75;"
        self.assertEqual(parse_input_max_count(content), 50)


class TestParseByIdx(TestCase):
    def test_normal_case(self):
        expected = [
            "IN_TENSOR_INPUT_IDS",
            "IN_TENSOR_POSITION_IDS",
            "IN_TENSOR_COS_TABLE",
            "IN_TENSOR_SIN_TABLE",
            "IN_TENSOR_ATTENTION_MASK",
            "IN_TENSOR_BLOCK_TABLES",
            "IN_TENSOR_SLOTS",
            "IN_TENSOR_LAYER_IDX",
            "IN_TENSOR_TOKEN_OFFSET",
            "IN_TENSOR_PLACE_HOLDER",
        ]
        content = parse_by_idx(MODEL_CASE_01)
        self.assertEqual(content, expected)

    def test_no_match(self):
        content = "No match here."
        self.assertEqual(parse_by_idx(content), [])

    def test_empty_string(self):
        content = ""
        self.assertEqual(parse_by_idx(content), [])


class TestGetATBModelNames(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.files = []

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_temp_file(self, content):
        temp_file = tempfile.NamedTemporaryFile(dir=self.temp_dir.name, delete=False, mode="w+")
        temp_file.write(content)
        temp_file.close()
        self.files.append(temp_file.name)
        return temp_file.name

    def test_single_file_single_model(self):
        file1 = self.create_temp_file("REGISTER_MODEL(ModelA);")
        result = get_atb_model_names([file1])
        self.assertEqual(result, "ModelA")

    def test_multiple_files_single_model(self):
        file1 = self.create_temp_file("REGISTER_MODEL(ModelA);")
        file2 = self.create_temp_file("REGISTER_MODEL(ModelB);")
        result = get_atb_model_names([file1, file2])
        self.assertEqual(result, "ModelA")

    def test_single_file_multiple_models(self):
        file1 = self.create_temp_file("REGISTER_MODEL(ModelC, ModelD);")
        result = get_atb_model_names([file1])
        self.assertEqual(result, "ModelC_ModelD")

    def test_no_match(self):
        file1 = self.create_temp_file("No match here.")
        result = get_atb_model_names([file1])
        self.assertEqual(result, "DecoderModel")

    def test_empty_file(self):
        file1 = self.create_temp_file("")
        result = get_atb_model_names([file1])
        self.assertEqual(result, "DecoderModel")

    def test_invalid_content(self):
        file1 = self.create_temp_file("REGISTER_MODEL();")
        result = get_atb_model_names([file1])
        self.assertEqual(result, "DecoderModel")

    def test_multiple_files_partial_match(self):
        file1 = self.create_temp_file("REGISTER_MODEL(ModelE);")
        file2 = self.create_temp_file("No match here.")
        result = get_atb_model_names([file1, file2])
        self.assertEqual(result, "ModelE")


class TestProcessLayer(TestCase):
    def test_process_layer_with_children(self):
        model = SimpleModel_3()
        result = process_layer("layer", model)
        self.assertEqual(result["name"], "layer")
        self.assertIn("attention", result)
        self.assertIn("input_layernorm", result)
        self.assertIn("post_attention_layernorm", result)
        attention_result = result["attention"]
        self.assertEqual(attention_result["name"], "attention")
        self.assertEqual(result["input_layernorm"]["name"], "layer1")
        self.assertEqual(result["post_attention_layernorm"]["name"], "norm2")

    def test_invalid_input(self):
        with self.assertRaises(AttributeError):
            process_layer("layer", "not a module")
