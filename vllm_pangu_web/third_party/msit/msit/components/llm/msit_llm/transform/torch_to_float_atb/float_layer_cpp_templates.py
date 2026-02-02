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

CPP_COPYRIGHT_HEADER = """/*
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


BASIC_CLASS_FORMATTER = """
namespace atb_speed {{
namespace {model_name_lower} {{

static const uint64_t IN_TENSOR_COUNT = 63;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 3;
static const uint64_t NODE_COUNT = 4;

{decoder_layer}

DecoderLayerBinder::DecoderLayerBinder() {{}}

DecoderLayerBinder::~DecoderLayerBinder() {{}}

{post_properties}

}} // namespace {model_name_lower}
}} // namespace atb_speed
"""

DECODER_LAYER_FORMATTER = """
atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation)
{{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";

    size_t nodeId = 0;
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpParallelNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    {attention_formatter}

    {residual_add_formatter}

    {mlp_formatter}

    {mlp_residual_add_formatter}

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {{
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    }};

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}}

"""


ATTENTION_FORMATTER = """
    // attention
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    // QKV linear param
    fusionAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    fusionAttentionParam.isBF16 = param.isBF16;
    fusionAttentionParam.qkvHasBias = true;
    fusionAttentionParam.layerLinearQuantType = param.linearQuantType;
    fusionAttentionParam.layerLinearTransposeType = param.linearTransposeType;
    fusionAttentionParam.packQuantType = param.packQuantType[0];
    fusionAttentionParam.supportLcoc = param.supportLcoc;
    fusionAttentionParam.enableLogN = param.enableLogN;  // for long sequence of qwen1
    atb::infer::RmsNormParam attenRmsNormParam;
    attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormParam.normParam.epsilon = param.rmsNormEps;
    fusionAttentionParam.normParamType = attenRmsNormParam;
    atb::infer::RmsNormParam attenRmsNormQuantParam;
    attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
    attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    fusionAttentionParam.normQuantParamType = attenRmsNormQuantParam;
    // rope param
    fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::ALL_ROTARY;
    fusionAttentionParam.ropeParam.rotaryCoeff = 2;
    // self attention param
    fusionAttentionParam.isFA = param.isFA;
    fusionAttentionParam.isPrefill = param.isPrefill;
    fusionAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    fusionAttentionParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    if (param.hiddenSizePerAttentionHead == 0) {{
        return atb::ERROR_INVALID_GRAPH;
    }}
    fusionAttentionParam.selfAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    fusionAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
    if (param.isFA) {{
        fusionAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    }} else {{
        fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }}
    fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    fusionAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    if (param.isBF16) {{
        fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_ALIBI;
    }} else {{
        fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::UNDEFINED;
    }}
    fusionAttentionParam.selfOutLinearTensorParallelInfo = {{param.rank, param.worldSize, param.backend}};
    Attention(fusionAttentionParam, &attentionNode.operation);
    attentionNode.inTensorIds = {{
        IN_HIDDEN_STATES,                                           // IN_HIDDEN_STATES
        IN_NORM_WEIGHT,                                             // IN_INPUT_NORM_WEIGHT
        IN_NORM_BIAS,                                               // IN_INPUT_NORM_BIAS
        IN_NORM_NEW_WEIGHT,                                         // IN_INPUT_NORM_NEW_WEIGHT
        IN_NORM_NEW_BIAS,                                           // IN_INPUT_NORM_NEW_BIAS
        IN_Q_WEIGHT,                                                // IN_QKV_WEIGHT_0
        IN_Q_SCALE,                                                 // IN_QKV_SCALE_0
        IN_Q_OFFSET,                                                // IN_QKV_OFFSET_0
        IN_Q_DEQSCALE,                                              // IN_QKV_DESCALE_0
        IN_Q_BIAS,                                                  // IN_QKV_DEOFFSET_0（quant场景下为quant_bias，非quant场景下为bias）
        IN_Q_COMPRESS_IDX,
        IN_K_WEIGHT,                                                // IN_QKV_WEIGHT_1
        IN_K_SCALE,                                                 // IN_QKV_SCALE_1
        IN_K_OFFSET,                                                // IN_QKV_OFFSET_1
        IN_K_DEQSCALE,                                              // IN_QKV_DESCALE_1
        IN_K_BIAS,                                                  // IN_QKV_DEOFFSET_1（quant场景下为quant_bias，非quant场景下为bias）
        IN_K_COMPRESS_IDX,
        IN_V_WEIGHT,                                                // IN_QKV_WEIGHT_2
        IN_V_SCALE,                                                 // IN_QKV_SCALE_2
        IN_V_OFFSET,                                                // IN_QKV_OFFSET_2
        IN_V_DEQSCALE,                                              // IN_QKV_DESCALE_2
        IN_V_BIAS,                                                  // IN_QKV_DEOFFSET_2（quant场景下为quant_bias，非quant场景下为bias）
        IN_V_COMPRESS_IDX,
        IN_COSEMBED,                                                // IN_COS_TABLE
        IN_SINEMBED,                                                // IN_SIN_TABLE
        IN_SEQ_LENGTHS,                                             // IN_SEQ_LEN
        IN_K_CACHE,                                                 // IN_K_CACHE
        IN_V_CACHE,                                                 // IN_V_CACHE
        IN_ATTENTIONMASK,                                           // IN_ATTENTION_MASK
        IN_TOKEN_OFFSET,                                            // IN_TOKEN_OFFSET
        IN_LAYER_ID,                                                // IN_LAYER_ID
        IN_BLOCK_TABLES,                                            // IN_BLOCK_TABLES
        IN_SLOTS,                                                   // IN_SLOTS
        IN_ATTENTION_OUT_WEIGHT,                                    // IN_ATTENTION_OUT_WEIGHT
        IN_ATTENTION_OUT_SCALE,                                     // IN_ATTENTION_OUT_SCALE
        IN_ATTENTION_OUT_OFFSET,                                    // IN_ATTENTION_OUT_OFFSET
        IN_ATTENTION_OUT_DEQSCALE,                                  // IN_ATTENTION_OUT_DESCALE
        IN_ATTENTION_OUT_BIAS,                                      // IN_ATTENTION_OUT_DEOFFSET（quant场景下为quant_bias，非quant场景下为bias）
        IN_ATTENTION_OUT_COMPRESS_IDX}};
    attentionNode.outTensorIds = {{INTERNAL_ATTENTIONOUT}};
"""


RESIDUAL_ADD_FORMATTER = """
    // residual
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {{
        IN_HIDDEN_STATES,
        INTERNAL_ATTENTIONOUT
    }};
    selfResidualAddNode.outTensorIds = {{INTERNAL_ATTENTIONRESIDUALADDOUT}};
"""


MLP_FORMATTER = """
    // mlp
    atb_speed::common::MlpParam<atb::infer::RmsNormParam> mlpParam;
    mlpParam.isBF16 = param.isBF16;
    mlpParam.layerLinearQuantType = param.linearQuantType;
    mlpParam.layerLinearTransposeType = param.linearTransposeType;
    mlpParam.packQuantType = param.packQuantType[1];
    mlpParam.supportLcoc = param.supportLcoc;
    // w2_w1(gate_up)
    mlpParam.mlpPackType = atb_speed::common::GATE_UP_WEIGHT_PACK;
    atb::infer::RmsNormParam mlpRmsNormParam;
    mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    mlpRmsNormParam.normParam.epsilon = param.rmsNormEps;
    mlpParam.normParamType = mlpRmsNormParam;
    atb::infer::RmsNormParam mlpRmsNormQuantParam;
    mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    mlpRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
    mlpRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    mlpParam.normQuantParamType = mlpRmsNormQuantParam;
    // c_proj(down)
    mlpParam.downLinearTensorParallelInfo = {{param.rank, param.worldSize, param.backend}};
    if (param.supportSwiGLU) {{
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
        mlpParam.activationParam.dim = -1;
        MlpSwiGLU(mlpParam, &mlpParallelNode.operation);
    }} else {{
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
        Mlp(mlpParam, &mlpParallelNode.operation);
    }}
    mlpParallelNode.inTensorIds = {{
        INTERNAL_ATTENTIONRESIDUALADDOUT,                             // INTERMEDIATE_RESIDUAL_ADD_OUT
        IN_SELFOUT_NORM_WEIGHT,                                       // IN_ATTENTION_NORM_WEIGHT
        IN_SELFOUT_NORM_BIAS,                                         // IN_ATTENTION_NORM_BIAS
        IN_SELFOUT_NORM_NEW_WEIGHT,                                   // IN_ATTENTION_NORM_NEW_WEIGHT
        IN_SELFOUT_NORM_NEW_BIAS,                                     // IN_ATTENTION_NORM_NEW_BIAS
        IN_MLP_W2_WEIGHT,                                             // IN_MLP_WEIGHT_0
        IN_MLP_W2_SCALE,                                              // IN_MLP_SCALE_0
        IN_MLP_W2_OFFSET,                                             // IN_MLP_OFFSET_0
        IN_MLP_W2_DEQSCALE,                                           // IN_MLP_DESCALE_0
        IN_MLP_W2_BIAS,                                               // IN_MLP_DEOFFSET_0
        IN_MLP_W2_COMPRESS_IDX,
        IN_MLP_W1_WEIGHT,                                             // IN_MLP_WEIGHT_1
        IN_MLP_W1_SCALE,                                              // IN_MLP_SCALE_1
        IN_MLP_W1_OFFSET,                                             // IN_MLP_OFFSET_1
        IN_MLP_W1_DEQSCALE,                                           // IN_MLP_DESCALE_1
        IN_MLP_W1_BIAS,                                               // IN_MLP_DEOFFSET_1
        IN_MLP_W1_COMPRESS_IDX,
        IN_MLP_CPROJ_WEIGHT,                                          // IN_MLP_DOWN_WEIGHT
        IN_MLP_CPROJ_SCALE,                                           // IN_MLP_DOWN_SCALE
        IN_MLP_CPROJ_OFFSET,                                          // IN_MLP_DOWN_OFFSET
        IN_MLP_CPROJ_DEQSCALE,                                        // IN_MLP_DOWN_DESCALE
        IN_MLP_CPROJ_BIAS,                                            // IN_MLP_DOWN_DEOFFSET
        IN_MLP_CPROJ_COMPRESS_IDX}};
    mlpParallelNode.outTensorIds = {{INTERNAL_MLPOUT}};
"""


MLP_RESIDUAL_ADD_FORMATTER = """
    // residual
    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {{
        INTERNAL_ATTENTIONRESIDUALADDOUT,
        INTERNAL_MLPOUT
    }};
    mlpResidualAddNode.outTensorIds = {{OUT_LAYEROUT}};
"""


PARSE_PARAM_FORMATTER = """
void DecoderLayerBinder::ParseParam(const nlohmann::json &paramJson)
{{
    ATB_LOG(INFO) << "enter DecoderLayerBinder ParseParam tokenOffset";
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {{
        tokenOffset_.push_back(item.get<int>());
    }}
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {{
        seqLen_.push_back(item.get<int>());
    }}
}}
"""

BIND_PARAM_HOST_TENSOR_FORMATTER = """
void DecoderLayerBinder::BindTensor(atb::VariantPack &variantPack)
{{
    ATB_LOG(INFO) << "enter DecoderLayerBinder BindTensor";
    variantPack.inTensors.at(IN_SEQ_LENGTHS).hostData = seqLen_.data();
    variantPack.inTensors.at(IN_TOKEN_OFFSET).hostData = tokenOffset_.data();
}}
"""
