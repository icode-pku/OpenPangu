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

COPYRIGHT_HEADER = """/*
 * Copyright (c) Huawei Technologies Co., Ltd. {year}. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * {licenses_url}
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
"""

INCLUDE_HEADER_FORMATTER = """
#ifndef ATB_SPEED_MODELS_{model_name_upper}_DECODER_LAYER_H
#define ATB_SPEED_MODELS_{model_name_upper}_DECODER_LAYER_H

#include <vector>
#include "nlohmann/json.hpp"

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
"""

BASIC_CLASS_FORMATTER = """
namespace atb_speed {{
namespace {model_name_lower} {{

{STRUCT_PARAM_FORMATTER}

{DECODER_LAYER_TENSOR_ID_FORMATTER}

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation);

class DecoderLayerBinder : public HostTensorBinder {{
public:
    DecoderLayerBinder();
    virtual ~DecoderLayerBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    int32_t layerId_ = 0;
}};


}}  // namespace {model_name_lower}
}}  // namespace atb_speed
#endif
"""

STRUCT_PARAM_FORMATTER = """
    struct DecoderLayerParam {{
        bool isFA = false;
        bool isPrefill = false;
        bool isBF16 = false;
        bool isPack = true;
        bool supportSwiGLU = false;
        int quantType = 0;
        float rmsNormEps = 0;
        int numAttentionHeadsPerRank = 0;
        int hiddenSizePerAttentionHead = 0;
        int numKeyValueHeadsPerRank = 0;
        bool supportLcoc = false;
        int rank = 0;
        int worldSize = 1;
        bool enableLogN = false;
        std::string backend = "hccl";
        std::vector<int> seqLen;
        std::vector<int> tokenOffset;
        std::vector<int> packQuantType = {{}};  // 两个元素，第一个元素代表QKV pack的量化类型，第二个元素代表MLP pack的量化类型
        // 七个元素，分别代表q，k，v，self attention out，gate，up，down linear的类型
        std::vector<int> linearQuantType = {{}};
        std::vector<int> linearTransposeType;
    }};
"""


DECODER_LAYER_TENSOR_ID_FORMATTER = """
    enum DecoderLayerTensorId : int {{
        IN_HIDDEN_STATES = 0,

        IN_NORM_WEIGHT,  // weight
        IN_NORM_BIAS,  // bias
        IN_NORM_NEW_WEIGHT,  // new weight
        IN_NORM_NEW_BIAS,  // new bias

        IN_Q_WEIGHT,  // weight
        IN_Q_BIAS,  // bias
        IN_Q_DEQSCALE,  // deq_scale
        IN_Q_OFFSET,  // offset
        IN_Q_SCALE,  // scale
        IN_Q_COMPRESS_IDX,

        IN_K_WEIGHT,  // weight
        IN_K_BIAS,  // bias
        IN_K_DEQSCALE,  // deq_scale
        IN_K_OFFSET,  // offset
        IN_K_SCALE,  // scale
        IN_K_COMPRESS_IDX,

        IN_V_WEIGHT,  // weight
        IN_V_BIAS,  // bias
        IN_V_DEQSCALE,  // deq_scale
        IN_V_OFFSET,  // offset
        IN_V_SCALE,  // scale
        IN_V_COMPRESS_IDX,

        IN_ATTENTION_OUT_WEIGHT,  // weight
        IN_ATTENTION_OUT_BIAS,  // bias
        IN_ATTENTION_OUT_DEQSCALE,  // deq_scale
        IN_ATTENTION_OUT_OFFSET,  // offset
        IN_ATTENTION_OUT_SCALE,  // scale
        IN_ATTENTION_OUT_COMPRESS_IDX,

        IN_SELFOUT_NORM_WEIGHT,  // weight
        IN_SELFOUT_NORM_BIAS,  // bias
        IN_SELFOUT_NORM_NEW_WEIGHT,  // new weight
        IN_SELFOUT_NORM_NEW_BIAS,  // new bias

        IN_MLP_W2_WEIGHT,  // weight
        IN_MLP_W2_BIAS,  // bias
        IN_MLP_W2_DEQSCALE,  // deq_scale
        IN_MLP_W2_OFFSET,  // offset
        IN_MLP_W2_SCALE,  // scale
        IN_MLP_W2_COMPRESS_IDX,

        IN_MLP_W1_WEIGHT,  // weight
        IN_MLP_W1_BIAS,  // bias
        IN_MLP_W1_DEQSCALE,  // deq_scale
        IN_MLP_W1_OFFSET,  // offset
        IN_MLP_W1_SCALE,  // scale
        IN_MLP_W1_COMPRESS_IDX,

        IN_MLP_CPROJ_WEIGHT,  // weight
        IN_MLP_CPROJ_BIAS,  // bias
        IN_MLP_CPROJ_DEQSCALE,  // deq_scale
        IN_MLP_CPROJ_OFFSET,  // offset
        IN_MLP_CPROJ_SCALE,  // scale
        IN_MLP_CPROJ_COMPRESS_IDX,

        IN_COSEMBED,
        IN_SINEMBED,
        IN_ATTENTIONMASK,
        IN_K_CACHE,
        IN_V_CACHE,
        IN_SEQ_LENGTHS,
        IN_PLACE_HOLDER,
        IN_TOKEN_OFFSET,
        IN_LAYER_ID,
        IN_BLOCK_TABLES,
        IN_SLOTS,
        IN_Q_LEN,

        OUT_LAYEROUT,

        INTERNAL_ATTENTIONOUT,
        INTERNAL_ATTENTIONRESIDUALADDOUT,
        INTERNAL_MLPOUT,
    }};

"""