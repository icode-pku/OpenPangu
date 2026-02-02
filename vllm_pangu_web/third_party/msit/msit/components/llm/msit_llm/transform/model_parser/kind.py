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

from typing import List

from torch.nn import Module, Linear, Embedding, LayerNorm, GELU
from msit_llm.common.log import logger


def mname(module: Module):
    return module.__class__.__name__


def convert(name, module: Module):
    module_class = module.__class__
    module_name = module_class.__name__
    lowered = module_name.lower()

    if "rms" in lowered:
        return rms_norm(name, module)
    elif module_class is Linear:
        return linear(name, module)
    elif module_class is Embedding:
        return embedding(name, module)
    elif module_class is LayerNorm:
        return layernorm(name, module)
    elif "rotary" in lowered and "embedding" in lowered:
        return rope(name, module)
    else:
        return {"name": mname(module), "children": []}


def linear(name, module):
    return {
        "name": name, 
        "kind": "Linear",
        "in_features": module.in_features,
        "out_features": module.out_features,
        "bias": module.bias is not None
    }


def embedding(name, module):
    return {
        "name": name, 
        "kind": "Embedding",
        "num_embeddings": module.num_embeddings,
        "embedding_dim": module.embedding_dim,
        "padding_idx": module.padding_idx
    }


def attention(name, modules: List[Module], size: int):
    ret = {"name": name, }
    
    names = [n for n, m in modules]
    modules = [m for n, m in modules]
    names_iter = iter(names)
    if size == 5:
        [q, k, v, output, r] = modules
        ret["structure"] = "q-k-v-o-r"
        ret["q"] = linear(next(names_iter), q)
        ret["k"] = linear(next(names_iter), k)
        ret["v"] = linear(next(names_iter), v)
        ret["o"] = linear(next(names_iter), output)
        ret["rope"] = rope(next(names_iter), r)
    elif size == 4:
        [q, kv, output, r] = modules
        ret["structure"] = "q-kv-o-r"
        ret["q"] = linear(next(names_iter), q)
        ret["kv"] = linear(next(names_iter), kv)
        ret["o"] = linear(next(names_iter), output)
        ret["rope"] = rope(next(names_iter), r)
    elif size == 3:
        [w, output, r] = modules
        ret["structure"] = "w-o-r"
        ret["w"] = linear(next(names_iter), w)
        ret["o"] = linear(next(names_iter), output)
        ret["rope"] = rope(next(names_iter), r)
    elif size == 2:
        [w, output] = modules
        ret["structure"] = "w-o"
        ret["w"] = linear(next(names_iter), w)
        ret["o"] = linear(next(names_iter), output)
    else:
        logger.error("error linear size")

    return ret


def mlp(name, modules: List[Module]):
    ret = {"ff": []}

    for n, m in modules:
        if isinstance(m, Linear):
            ret["ff"].append(linear(n, m))
        else:
            ret["act"] = activation(n, m)

    return ret


def layernorm(name, module):
    return {
        "name": name, 
        "kind": "LayerNorm",
        "normalized_shape": module.normalized_shape[0],
        "eps": module.eps,
        "element_affine": module.elementwise_affine,
        "bias": module.bias is not None
    }


def rope(name, module: Module):
    return {
        "name": name, 
        "kind": "RotaryEmbedding",
        "base": module.base if hasattr(module, "base") else -1,
        "dim": module.dim if hasattr(module, "dim") else -1,
        "max_position_embeddings": module.max_position_embeddings if hasattr(module, "max_position_embeddings") else -1,
        "max_seq_len_cached": module.max_seq_len_cached if hasattr(module, "max_seq_len_cached") else -1
    }


def rms_norm(name, module: Module):
    ret = {"name": name, "kind": "RMSNorm"}
    eps_like = ["epsilon", "variance_epsilon", "eps"]

    for name in eps_like:
        if hasattr(module, name):
            ret["eps"] = getattr(module, name)
            break

    return ret


def activation(name, module: Module):
    module_class = module.__class__

    if module_class is GELU:
        return {
            "name": name,
            "kind": "GELU",
            "approximate": module.approximate == "tanh"
        }

    module_name = module_class.__name__
    lowered = module_name.lower()

    if "tanh" in lowered and "gelu" in lowered:
        return {
            "name": name,
            "kind": "GELU",
            "approximate": True
        }

    return {"name": name, "kind": module_name}
