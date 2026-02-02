# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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


__all__ = ["KnowledgeBase", "KnowledgeFactory", "Pattern"]


from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase
from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory

from auto_optimizer.pattern.knowledges import knowledge_conv1d2conv2d
from auto_optimizer.pattern.knowledges import knowledge_merge_consecutive_slice
from auto_optimizer.pattern.knowledges import knowledge_transpose_large_input_conv
from auto_optimizer.pattern.knowledges import knowledge_merge_consecutive_concat
from auto_optimizer.pattern.knowledges import knowledge_type_cast
from auto_optimizer.pattern.knowledges import knowledge_split_qkv_matmul
from auto_optimizer.pattern.knowledges import knowledge_split_large_kernel
from auto_optimizer.pattern.knowledges import knowledge_resize_mode_to_nearest
from auto_optimizer.pattern.knowledges import knowledge_topk_fix
from auto_optimizer.pattern.knowledges import knowledge_merge_casts
from auto_optimizer.pattern.knowledges import knowledge_empty_slice_fix
from auto_optimizer.pattern.knowledges import knowledge_dynamic_reshape
from auto_optimizer.pattern.knowledges import knowledge_gather_to_split
from auto_optimizer.pattern.knowledges import knowledge_avgpool_split
from auto_optimizer.pattern.knowledges import knowledge_bn_folding
from auto_optimizer.pattern.knowledges import knowledge_modify_reflection_pad

from auto_optimizer.pattern.pattern import Pattern