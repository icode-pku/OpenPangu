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
 */"""

all_atb_operation_headers = [
    "layers/operations/fusion_attention.h",
    "layers/operations/linear.h",
    "layers/operations/linear_parallel.h",
    "layers/operations/lmhead.h",
    "layers/operations/mlp.h",
    "layers/operations/mlp_swiglu.h",
    "layers/operations/norm_linear.h",
    "layers/operations/positional_embedding.h",
    "layers/operations/qkv_linear_split.h",
    "layers/operations/self_attention.h",
    "layers/operations/word_embedding.h",
]

INCLUDE_HEADER_FORMATTER = """
#include "nlohmann/json.hpp"
#include "vector"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "layers/operations/lmhead.h"
#include "layers/operations/word_embedding.h"
#include "layers/operations/positional_embedding.h"
{other_operations}
#include "models/{model_name_lower}/layer/decoder_layer.h"
#include "models/{model_name_lower}/model/decoder_model.h"
"""

BASIC_CLASS_FORMATTER = """
namespace atb_speed {{
namespace {model_name_lower} {{

{pre_properties}

DecoderModel::DecoderModel(const std::string &param) : Model("DecoderModel", param)
{{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}}

DecoderModel::~DecoderModel() {{}}

uint32_t DecoderModel::GetInputNum() const {{ return graph_.inTensors.size(); }}

uint32_t DecoderModel::GetOutputNum() const {{ return graph_.outTensors.size(); }}

{post_properties}

}} // namespace {model_name_lower}
}} // namespace atb_speed
"""

WEIGHT_COUNT_FORMATTER = """
// Weight count
const int WEIGHT_COUNT_PER_LAYER = 50;
const int WEIGHT_COUNT_WORD_EMBEDDINGNODE = 1;
const int WEIGHT_COUNT_POST_NORM = 1;
const int WEIGHT_COUNT_LM_HEAD = 1;
"""

OPERATION_COUNT_FORMATTER = """
// Operation count
const int OPERATION_COUNT_BEFORE_LAYER = 2;  // wte(wordEmbed) + gather(cos/sin embedding)
const int OPERATION_COUNT_AFTER_LAYER = 2;  // RmsNorm + LmHead
"""

IN_TENSOR_ID_FORMATTER = """
enum InTensorId : int {{
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
}};
"""

INTERNAL_TENSOR_ID_FORMATTER = """
enum InternalTensorId : int {{
    // define internelTensor
    // idx: 0, shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
    INTERNEL_TENSOR_HIDDEN_STATES,
    // idx: 1, shape: [batchSize * seqLen, hiddenSizePerAttentionHead]
    INTERNEL_TENSOR_COS_EMB,
    // idx: 2, shape: [batchSize * seqLen, hiddenSizePerAttentionHead]
    INTERNEL_TENSOR_SIN_EMB,
    INTERNAL_TENSOR_MAX,
}};
"""

OUT_TENSOR_ID_FORMATTER = """
enum OutTensorId : int {{
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
}};
"""

FROM_STRING_FORMATTER = """
void DecoderModel::Param::FromString(const std::string &param)
{{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    isFA = paramJson["isFA"].get<bool>();
    isPrefill = paramJson["isPrefill"].get<bool>();
    isBF16 = paramJson["isBF16"].get<bool>();
    if (paramJson.contains("withEmbedding")) {{
        withEmbedding = paramJson["withEmbedding"].get<bool>();
    }}
    if (paramJson.contains("isEmbeddingParallel")) {{
        isEmbeddingParallel = paramJson["isEmbeddingParallel"].get<bool>();
    }}
    if (paramJson.contains("isLmHeadParallel")) {{
        isLmHeadParallel = paramJson["isLmHeadParallel"].get<bool>();
    }}
    if (paramJson.contains("lmHeadTransposeType")) {{
        lmHeadTransposeType = paramJson["lmHeadTransposeType"].get<int>();
    }}
    if (paramJson.contains("supportSwiGLU")) {{
        supportSwiGLU = paramJson["supportSwiGLU"].get<bool>();
    }}
    if (paramJson.contains("rmsNormEps")) {{
        rmsNormEps = paramJson["rmsNormEps"].get<float>();
    }}
    if (paramJson.contains("numAttentionHeadsPerRank")) {{
        numAttentionHeadsPerRank = paramJson["numAttentionHeadsPerRank"].get<int>();
    }}
    if (paramJson.contains("hiddenSizePerAttentionHead")) {{
        hiddenSizePerAttentionHead = paramJson["hiddenSizePerAttentionHead"].get<int>();
    }}
    if (paramJson.contains("numHiddenLayers")) {{
        numHiddenLayers = paramJson["numHiddenLayers"].get<int>();
    }}
    if (paramJson.contains("layerNum")) {{
        numHiddenLayers = paramJson["layerNum"].get<int>();
    }}
    if (paramJson.contains("numKeyValueHeadsPerRank")) {{
        numKeyValueHeadsPerRank = paramJson["numKeyValueHeadsPerRank"].get<int>();
    }}
    if (paramJson.contains("supportLcoc")) {{
        supportLcoc = paramJson["supportLcoc"].get<bool>();
    }}
    if (paramJson.contains("rank")) {{
        rank = paramJson["rank"].get<int>();
    }}
    if (paramJson.contains("rankSize")) {{
        worldSize = paramJson["rankSize"].get<int>();
    }}
    if (paramJson.contains("worldSize")) {{
        worldSize = paramJson["worldSize"].get<int>();
    }}
    if (paramJson.contains("backend")) {{
        backend = paramJson["backend"].get<std::string>();
    }}
    if (paramJson.contains("enableLogN")) {{
        enableLogN = paramJson["enableLogN"].get<bool>();
    }}

    for (auto item : paramJson["tokenOffset"]) {{
        tokenOffset.push_back(item.get<int>());
    }}
    for (auto item : paramJson["seqLen"]) {{
        seqLen.push_back(item.get<int>());
    }}
    for (auto item : paramJson["packQuantType"]) {{
        packQuantType.push_back(item.get<std::vector<int>>());
    }}
    for (auto item : paramJson["linearQuantType"]) {{
        linearQuantType.push_back(item.get<std::vector<int>>());
    }}
    for (auto item : paramJson["linearTransposeType"]) {{
        linearTransposeType.push_back(item.get<std::vector<int>>());
    }}
    ATB_LOG(INFO) << "DecoderModel param" << ", isFA:" << isFA << ", isPrefill:" << isPrefill
                  << ", isBF16:" << isBF16
                  << ", withEmbedding: " << withEmbedding
                  << ", isEmbeddingParallel: " << isEmbeddingParallel
                  << ", isLmHeadParallel: " << isLmHeadParallel << ", supportSwiGLU: " << supportSwiGLU
                  << ", rmsNormEps:" << rmsNormEps << ", numAttentionHeadsPerRank:" << numAttentionHeadsPerRank
                  << ", hiddenSizePerAttentionHead:" << hiddenSizePerAttentionHead
                  << ", numHiddenLayers:" << numHiddenLayers
                  << ", numKeyValueHeadsPerRank:" << numKeyValueHeadsPerRank
                  << ", supportLcoc:" << supportLcoc << ", rank:" << rank << ", worldSize:" << worldSize
                  << ", backend:" << backend << ", tokenOffset:" << tokenOffset << ", seqLen:" << seqLen;
}}
"""

INFER_SHAPE_FORMATTER = """
atb::Status DecoderModel::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs
)
{{
    ATB_LOG(INFO) << "Enter DecoderModel InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {{
        return atb::ERROR_INVALID_GRAPH;
    }}

    const int64_t vocabSizePerRank = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    // FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSisze]
    outTensorDescs.at(0).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
    outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum + (param_.withEmbedding ? 1 : 0);
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];

    if (param_.isFA) {{  // unpadInputs = false
        outTensorDescs.at(0).shape.dims[1] =
            param_.isPrefill ? inTensorDescs.at(graph_.inTensors.size() - 1).shape.dims[0] : 1;
    }} else {{  // unpadInputs = true
        if (param_.isPrefill) {{
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(graph_.inTensors.size() - 1).shape.dims[0];
        }}
    }}

    if (param_.isLmHeadParallel) {{
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank * param_.worldSize;
    }} else {{
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank;
    }}

    return atb::NO_ERROR;
}}
"""

PARSE_PARAM_FORMATTER = """
atb::Status DecoderModel::ParseParam(const std::string &param)
{{
    ATB_LOG(INFO) << "ParseParam start.";
    nlohmann::json paramJson = nlohmann::json::parse(param);

    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {{
        tokenOffset_.push_back(item.get<int>());
        ATB_LOG(INFO) << "tokenOffset value: " << item;
    }}
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {{
        seqLen_.push_back(item.get<int>());
        ATB_LOG(INFO) << "Prefill" << paramJson["isPrefill"] << "seqLen value: " << item;
    }}
    ATB_LOG(INFO) << "ParseParam end.";
    return atb::NO_ERROR;
}}
"""

BIND_PARAM_HOST_TENSOR_FORMATTER = """
atb::Status DecoderModel::BindParamHostTensor(uint32_t nodeId)
{{
    ATB_LOG(INFO) << "BindParamHostTensor";
    ATB_LOG(INFO) << "nodeId = " << nodeId;

    auto upperBound = (param_.withEmbedding ? OPERATION_COUNT_BEFORE_LAYER : OPERATION_COUNT_BEFORE_LAYER - 1);
    auto lowerBound = upperBound + param_.numHiddenLayers;
    if (nodeId < static_cast<uint32_t>(upperBound) || nodeId >= static_cast<uint32_t>(lowerBound)) {{
        return atb::NO_ERROR;
    }}

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t tokenOffsetTensorId = DecoderLayerTensorId::IN_TOKEN_OFFSET;
    const uint32_t seqLenTensorId = DecoderLayerTensorId::IN_SEQ_LENGTHS;
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}}
"""

BUILD_GRAPH_FORMATTER = """
int64_t DecoderModel::BuildGraph()
{{
    // set size
    const int weightTensorSize = (param_.withEmbedding ? WEIGHT_COUNT_WORD_EMBEDDINGNODE : 0) +
                                 WEIGHT_COUNT_PER_LAYER * param_.numHiddenLayers + WEIGHT_COUNT_POST_NORM +
                                 WEIGHT_COUNT_LM_HEAD;

    graph_.weightTensors.resize(weightTensorSize);

    graph_.kCacheTensors.resize(param_.numHiddenLayers);
    graph_.vCacheTensors.resize(param_.numHiddenLayers);

    graph_.inTensors.resize(IN_TENSOR_MAX - 1);
    graph_.outTensors.resize(OUT_TENSOR_MAX);
    graph_.internalTensors.resize(INTERNAL_TENSOR_MAX);

    const int nodeSize = param_.numHiddenLayers +
                         (param_.withEmbedding ? OPERATION_COUNT_BEFORE_LAYER : OPERATION_COUNT_BEFORE_LAYER - 1) +
                         OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    ATB_LOG(INFO) << "weightTensors.size=" << graph_.weightTensors.size()
                  << ", inTensors.size=" << graph_.inTensors.size()
                  << ", outTensors.size=" << graph_.outTensors.size()
                  << ", internalTensor.size=" << graph_.internalTensors.size()
                  << ", nodes.size=" << graph_.nodes.size();

    ATB_LOG(INFO) << "DecoderModel build graph begin";
    int nodeId = 0;

    atb::Operation *op = nullptr;

    {BUILD_GRAPH_PRE_PROCESS_FORMATTER}
    {BUILD_GRAPH_PRE_PROCESS_NORM_FORMATTER}
    {BUILD_GRAPH_LAYERS_FORMATTER}
    {BUILD_GRAPH_POST_PROCESS_NORM_FORMATTER}
    {BUILD_GRAPH_POST_PROCESS_LMHEAD_FORMATTER}
    ATB_LOG(INFO) << "DecoderModel build graph success";
    return atb::NO_ERROR;
}}
"""

BUILD_GRAPH_PRE_PROCESS_FORMATTER = """
    // wte
    if (param_.withEmbedding) {{
        auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
        atb_speed::common::WordEmbeddingParam wordEmbeddingParam;
        wordEmbeddingParam.unpadInputs = !param_.isFA;
        if (param_.isEmbeddingParallel) {{
            wordEmbeddingParam.tensorParallelInfo = {{param_.rank, param_.worldSize, param_.backend}};
        }};
        atb_speed::common::WordEmbedding(wordEmbeddingParam, &op);
        wordEmbeddingNode.operation.reset(op);
        wordEmbeddingNode.inTensors = {{&graph_.weightTensors.at(0),  // shape: [vocabSize + 1, hiddenSize]
            &graph_.inTensors.at(IN_TENSOR_INPUT)}};
        wordEmbeddingNode.outTensors = {{&graph_.internalTensors.at(INTERNEL_TENSOR_HIDDEN_STATES)}};
        ATB_LOG(INFO) << "[+] wordEmbeddingNode";
    }}

    // gather
    auto &peGatherNode = graph_.nodes.at(nodeId++);
    atb_speed::common::PositionalEmbeddingGather(&op);
    peGatherNode.operation.reset(op);
    peGatherNode.inTensors = {{
        &graph_.inTensors.at(IN_TENSOR_POSITION_IDS),
        &graph_.inTensors.at(IN_TENSOR_COS_TABLE),
        &graph_.inTensors.at(IN_TENSOR_SIN_TABLE),
    }};
    peGatherNode.outTensors = {{
        &graph_.internalTensors.at(INTERNEL_TENSOR_COS_EMB),
        &graph_.internalTensors.at(INTERNEL_TENSOR_SIN_EMB)
    }};
    ATB_LOG(INFO) << "[+] peGatherNode";

    atb::Tensor *firstInTensor = param_.withEmbedding ? &graph_.internalTensors.at(INTERNEL_TENSOR_HIDDEN_STATES)
                                                      : &graph_.inTensors.at(IN_TENSOR_INPUT);

"""

BUILD_GRAPH_PRE_PROCESS_NORM_FORMATTER = """
"""

BUILD_GRAPH_LAYERS_FORMATTER = """
    // layers
    for (int layerId = 0; layerId < param_.numHiddenLayers; ++layerId) {{
        auto &layerNode = graph_.nodes.at(nodeId++);
        atb_speed::{model_name_lower}::DecoderLayerParam layerParam;
        layerParam.isFA = param_.isFA;
        layerParam.isPrefill = param_.isPrefill;
        layerParam.isBF16 = param_.isBF16;
        layerParam.supportSwiGLU = param_.supportSwiGLU;
        layerParam.packQuantType = param_.packQuantType[layerId];
        layerParam.linearQuantType = param_.linearQuantType[layerId];
        layerParam.linearTransposeType = param_.linearTransposeType[layerId];
        layerParam.supportLcoc = param_.supportLcoc;
        layerParam.rmsNormEps = param_.rmsNormEps;
        layerParam.numAttentionHeadsPerRank = param_.numAttentionHeadsPerRank;
        layerParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
        layerParam.numKeyValueHeadsPerRank = param_.numKeyValueHeadsPerRank;
        layerParam.rank = param_.rank;
        layerParam.worldSize = param_.worldSize;
        layerParam.backend = param_.backend;
        layerParam.enableLogN = param_.enableLogN;
        atb_speed::{model_name_lower}::DecoderLayer(layerParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {{
            layerNode.inTensors.at(inTensorId++) =
                &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId +
                                         (param_.withEmbedding ? WEIGHT_COUNT_WORD_EMBEDDINGNODE : 0));
        }}
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNEL_TENSOR_COS_EMB);
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNEL_TENSOR_SIN_EMB);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTION_MASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQ_LENGTHS);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_PLACEHOLDER);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKEN_OFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_KV_CACHE_IDX);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(param_.supportSpeculate ? IN_TENSOR_Q_LEN : IN_PLACEHOLDER);

        layerNode.outTensors = {{&graph_.internalTensors.at(INTERNEL_TENSOR_HIDDEN_STATES)}};
        ATB_LOG(INFO) << "[+] layerNode_" << layerId;
        firstInTensor = layerNode.outTensors.at(0);
    }}

"""

BUILD_GRAPH_POST_PROCESS_NORM_FORMATTER = """
    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - WEIGHT_COUNT_POST_NORM - WEIGHT_COUNT_LM_HEAD;
    finalNormNode.inTensors = {{firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)}};
    finalNormNode.outTensors = {{
        // shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
        &graph_.internalTensors.at(INTERNEL_TENSOR_HIDDEN_STATES)
    }};
    ATB_LOG(INFO) << "[+] finalNormNode";

"""

BUILD_GRAPH_POST_PROCESS_LMHEAD_FORMATTER = """
    auto &lmHeadNode = graph_.nodes.at(nodeId++);
    atb_speed::common::LmHeadParam lmHeadParam;
    lmHeadParam.unpadInputs = !param_.isFA;
    lmHeadParam.gatherAhead = param_.isPrefill;
    lmHeadParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
    lmHeadParam.linearParallelParam.fusionLinearParam.isBF16 = param_.isBF16;
    lmHeadParam.linearParallelParam.unpadInputs = !param_.isFA;
    lmHeadParam.linearParallelParam.fusionLinearParam.transposeType = param_.lmHeadTransposeType;
    if (param_.isLmHeadParallel) {{
        lmHeadParam.linearParallelParam.parallelType = atb_speed::common::COLUMN_PARALLEL;
        lmHeadParam.linearParallelParam.tensorParallelInfo.rank = param_.rank;
        lmHeadParam.linearParallelParam.tensorParallelInfo.worldSize = param_.worldSize;
        lmHeadParam.linearParallelParam.tensorParallelInfo.backend = param_.backend;
    }}
    LmHead(lmHeadParam, &op);
    lmHeadNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD;
    lmHeadNode.inTensors = {{
        &graph_.internalTensors.at(INTERNEL_TENSOR_HIDDEN_STATES),
        // shape: [vocabSizePerRank, hiddenSize]
        &graph_.weightTensors.at(finalLinearWeightTensorId),
        // LmHead not quantized, using placeholder for weights
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_TENSOR_LOGTIS_INDICES)
    }};
    // shpae: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode.outTensors = {{&graph_.outTensors.at(0)}};
    ATB_LOG(INFO) << "[+] lmHeadNode";

"""