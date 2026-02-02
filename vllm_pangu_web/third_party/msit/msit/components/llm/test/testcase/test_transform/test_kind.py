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

from unittest import TestCase

from torch.nn import Linear, Embedding, LayerNorm, SiLU, GELU, Module

from msit_llm.transform.model_parser.kind import convert, attention, mlp, activation


class RotaryEmbedding(Module):
    def __init__(
            self,
            dim: int,
            base: int = 10000,
            max_position_embeddings: int = 2048,
            max_seq_len_cached: int = 2048
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len_cached = max_seq_len_cached


class RMSNorm(Module):
    def __init__(self, eps: float):
        super().__init__()
        self.variance_epsilon = eps


class Attention(Module):
    def __init__(self):
        super().__init__()
        self.q_proj = Linear(16, 16, bias=True)
        self.k_proj = Linear(16, 16, bias=True)
        self.v_proj = Linear(16, 16, bias=True)
        self.o_proj = Linear(16, 16, bias=False)

        self.rotary_emb = RotaryEmbedding(
            dim=16,
            base=100,
            max_position_embeddings=100,
            max_seq_len_cached=100
        )


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = Linear(16, 16, bias=False)
        self.up_proj = Linear(16, 16, bias=False)
        self.down_proj = Linear(16, 16, bias=True)
        self.act_fn = SiLU()


class TestKind(TestCase):
    def test_linear(self):
        linear = Linear(16, 48, False)
        self.assertEqual(
            {   
                "name": "linear",
                "kind": "Linear",
                "in_features": 16,
                "out_features": 48,
                "bias": False
            },
            convert("linear", linear)
        )


    def test_embedding(self):
        embedding1 = Embedding(16, 16, 4)
        embedding2 = Embedding(16, 48)
        self.assertEqual(
            {
                "name": "embedding",
                "kind": "Embedding",
                "num_embeddings": 16,
                "embedding_dim": 16,
                "padding_idx": 4
            },
            convert('embedding', embedding1)
        )
        self.assertEqual(
            {
                "name": 'embedding',
                "kind": "Embedding",
                "num_embeddings": 16,
                "embedding_dim": 48,
                "padding_idx": None
            },
            convert('embedding', embedding2)
        )

    def test_attention(self):
        attn = Attention()
        sub = list(attn.named_children())

        self.assertEqual(
            {
                "name": 'attention',
                "structure": "q-k-v-o-r",
                "q": {
                    "name": "q_proj",
                    "kind": "Linear",
                    "in_features": 16,
                    "out_features": 16,
                    "bias": True
                },
                "k": {
                    "name": "k_proj",
                    "kind": "Linear",
                    "in_features": 16,
                    "out_features": 16,
                    "bias": True
                },
                "v": {
                    "name": "v_proj",
                    "kind": "Linear",
                    "in_features": 16,
                    "out_features": 16,
                    "bias": True
                },
                "o": {
                    "name": "o_proj",
                    "kind": "Linear",
                    "in_features": 16,
                    "out_features": 16,
                    "bias": False
                },
                "rope": {
                    "name": "rotary_emb",
                    "kind": "RotaryEmbedding",
                    "base": 100,
                    "dim": 16,
                    "max_position_embeddings": 100,
                    "max_seq_len_cached": 100
                }
            },
            attention('attention', sub, 5)
        )

    def test_mlp(self):
        my_mlp = MLP()
        sub = list(my_mlp.named_children())
        self.assertEqual(
            {
                "ff": [
                    {
                        "name": "gate_proj",
                        "kind": "Linear",
                        "in_features": 16,
                        "out_features": 16,
                        "bias": False
                    }, {
                        "name": "up_proj",
                        "kind": "Linear",
                        "in_features": 16,
                        "out_features": 16,
                        "bias": False
                    }, {
                        "name": "down_proj",
                        "kind": "Linear",
                        "in_features": 16,
                        "out_features": 16,
                        "bias": True
                    }
                ],
                "act": {"name": "act_fn", "kind": "SiLU"}
            },
            mlp("mlp", sub),
        )

    def test_layernorm(self):
        layernorm1 = LayerNorm(12)
        layernorm2 = LayerNorm(36, 0.2, False, False)

        self.assertEqual(
            {
                "name": "layernorm1",
                "kind": "LayerNorm",
                "normalized_shape": 12,
                "eps": 1e-5,
                "element_affine": True,
                "bias": True
            },
            convert("layernorm1", layernorm1)
        )
        self.assertEqual(
            {
                "name": "layernorm2",
                "kind": "LayerNorm",
                "normalized_shape": 36,
                "eps": 0.2,
                "element_affine": False,
                "bias": False
            },
            convert("layernorm2", layernorm2)
        )

    def test_rope(self):
        rope1 = RotaryEmbedding(dim=12)
        rope2 = RotaryEmbedding(dim=36, base=100, max_position_embeddings=100)

        self.assertEqual(
            {
                "name": "rope1",
                "kind": "RotaryEmbedding",
                "base": 10000,
                "dim": 12,
                "max_position_embeddings": 2048,
                "max_seq_len_cached": 2048
            },
            convert("rope1", rope1)
        )
        self.assertEqual(
            {   
                "name": "rope2",
                "kind": "RotaryEmbedding",
                "base": 100,
                "dim": 36,
                "max_position_embeddings": 100,
                "max_seq_len_cached": 2048
            },
            convert("rope2", rope2)
        )

    def test_rms_norm(self):
        rms_norm1 = RMSNorm(eps=0.2)
        rms_norm2 = RMSNorm(eps=1e-5)

        self.assertEqual(
            {
                "name": "rms_norm1",
                "kind": "RMSNorm",
                "eps": 0.2
            },
            convert("rms_norm1", rms_norm1)
        )
        self.assertEqual(
            {
                "name": "rms_norm2",
                "kind": "RMSNorm",
                "eps": 1e-5
            },
            convert("rms_norm2", rms_norm2)
        )

    def test_activation(self):
        silu = SiLU()
        gelu1 = GELU()
        gelu2 = GELU(approximate="tanh")

        self.assertEqual({"name": "SILU", "kind": "SiLU"}, activation("SILU", silu))
        self.assertEqual({"name": "GELU", "kind": "GELU", "approximate": False}, activation("GELU", gelu1))
        self.assertEqual({"name": "GELU", "kind": "GELU", "approximate": True}, activation("GELU", gelu2))
