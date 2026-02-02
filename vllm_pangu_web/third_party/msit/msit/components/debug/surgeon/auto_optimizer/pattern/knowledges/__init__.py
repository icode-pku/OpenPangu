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


from auto_optimizer.pattern.knowledges.knowledge_conv1d2conv2d import KnowledgeConv1d2Conv2d
from auto_optimizer.pattern.knowledges.knowledge_merge_consecutive_slice import KnowledgeMergeConsecutiveSlice
from auto_optimizer.pattern.knowledges.knowledge_transpose_large_input_conv import KnowledgeTransposeLargeInputConv
from auto_optimizer.pattern.knowledges.knowledge_merge_consecutive_concat import KnowledgeMergeConsecutiveConcat
from auto_optimizer.pattern.knowledges.knowledge_type_cast import KnowledgeTypeCast
from auto_optimizer.pattern.knowledges.knowledge_split_qkv_matmul import KnowledgeSplitQKVMatmul
from auto_optimizer.pattern.knowledges.knowledge_split_large_kernel import KnowledgeSplitLargeKernelConv
from auto_optimizer.pattern.knowledges.knowledge_resize_mode_to_nearest import KnowledgeResizeModeToNearest
from auto_optimizer.pattern.knowledges.knowledge_topk_fix import KnowledgeTopkFix
from auto_optimizer.pattern.knowledges.knowledge_merge_casts import KnowledgeMergeCasts
from auto_optimizer.pattern.knowledges.knowledge_empty_slice_fix import KnowledgeEmptySliceFix
from auto_optimizer.pattern.knowledges.knowledge_dynamic_reshape import KnowledgeDynamicReshape
from auto_optimizer.pattern.knowledges.knowledge_gather_to_split import KnowledgeGatherToSplit
from auto_optimizer.pattern.knowledges.knowledge_avgpool_split import KnowledgeAvgPoolSplit
from auto_optimizer.pattern.knowledges.knowledge_bn_folding import KnowledgeBNFolding
from auto_optimizer.pattern.knowledges.knowledge_modify_reflection_pad import KnowledgeModifyReflectionPad
from auto_optimizer.pattern.knowledges.big_kernel.knowledge_big_kernel import KnowledgeBigKernel
