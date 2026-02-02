export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

MODEL_PATH=$PROJECT_PATH/resource/msit_llm             #原模型路径

pip install $MODEL_PATH/pytorch_v2.1.0_py310/apex-0.1+ascend-cp310-cp310-linux_aarch64.whl --force-reinstall
pip install $MODEL_PATH/torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall
pip install $MODEL_PATH/pytorch_v2.1.0_py310/torch_npu-2.1.0.post7-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall

cd -

CUR_PATH=$PWD
TEST_DIR=${MODEL_NAME}_`date +%y%m%d%H%M`
ATB_FILE_NAME="flash_attention_rope_layer"
echo ""
echo ">>>> TEST_DIR=$TEST_DIR"
mkdir -p $TEST_DIR
cd $TEST_DIR

echo '
#ifndef BAICHUAN2_7B_FLASH_ATTENTION_ROPE_LAYER_H
#define BAICHUAN2_7B_FLASH_ATTENTION_ROPE_LAYER_H

#include <atb/atb_infer.h>
#include <atb/svector.h>

#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/str_split.h"

namespace atb_speed {
namespace baichuan2_7b {
struct FlashAttentionRopeLayerParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int rank = 0;
    int rankSize = 1;
    std::string backend = "hccl";
    std::string model = "baichuan2_7b";
};


atb::Status FlashAttentionRopeLayer(const FlashAttentionRopeLayerParam &param, atb::Operation **operation);
void from_json(const nlohmann::json &paramJson, FlashAttentionRopeLayerParam &param);
atb::Operation *CreateFlashAttentionRopeLayer(const nlohmann::json &paramJson);

class FlashAttentionRopeLayerBinder : public HostTensorBinder{
public:
    FlashAttentionRopeLayerBinder();
    ~FlashAttentionRopeLayerBinder() override;
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};
} // namespace baichuan2_7b
} // namespace atb_speed
#endif
' > ${ATB_FILE_NAME}.h

echo '
#include "flash_attention_rope_layer.h"

#include "layers/mlp_gate_v2.h"
#include "layers/parallel_layer_v2.h"
#include "models/baichuan2/7b/operation/rope.h"


namespace atb_speed {
namespace baichuan2_7b {
enum FlashAttentionRopeLayerTensorId : int {
    IN_HIDDENSTATES = 0,

    IN_NORMWEIGHT,
    IN_QKVMIXEDLINEARWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_MLPUPWEIGHT,

    IN_COS_EMBED, // 目前只支持FP16
    IN_SIN_EMBED,
    IN_ATTENTIONMASK,
    IN_PASTKEY,
    IN_PASTVALUE,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_HOLDER,
    IN_LAYERID,
    OUT_LAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_QKVMIXEDLINEAROUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 17;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 12;
static const uint64_t NODE_COUNT = 10;

void from_json(const nlohmann::json &paramJson, FlashAttentionRopeLayerParam &param)
{
    paramJson.at("rmsNormEps").get_to(param.rmsNormEps);
    paramJson.at("headNum").get_to(param.headNum);
    paramJson.at("dk").get_to(param.dk);
    if (paramJson.contains("rank")) {
        paramJson.at("rank").get_to(param.rank);
    }
    if (paramJson.contains("rankSize")) {
        paramJson.at("rankSize").get_to(param.rankSize);
    }
    if (paramJson.contains("backend")) {
        paramJson.at("backend").get_to(param.backend);
    }
}

atb::Operation *CreateFlashAttentionRopeLayer(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    atb::Operation *op;
    atb_speed::baichuan2_7b::FlashAttentionRopeLayer(paramJson.get<FlashAttentionRopeLayerParam>(), &op);
    return op;
}

atb::Status FlashAttentionRopeLayer(const FlashAttentionRopeLayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &qkvLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitQKVNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionKvCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOperation(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};



    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    CreateOperation(linearParam, &qkvLinearNode.operation);
    qkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXEDLINEARWEIGHT};
    qkvLinearNode.outTensorIds = {INTERMIDATE_QKVMIXEDLINEAROUT};

    atb::infer::SplitParam splitParam = {2, 3};
    CreateOperation(splitParam, &splitQKVNode.operation);
    splitQKVNode.inTensorIds = {INTERMIDATE_QKVMIXEDLINEAROUT};
    splitQKVNode.outTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_MIXEDV};

    atb_speed::baichuan2_7b::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 2;
    ropeParam.headNum = param.headNum;
    atb_speed::baichuan2_7b::Rope(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, IN_COS_EMBED, IN_SIN_EMBED, IN_SEQLEN};
    ropeNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK};

    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qkScale = 1.0 / sqrt(param.dk);
    selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    CreateOperation(selfAttentionParam, &selfAttentionKvCacheNode.operation);
    selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ,
                                            INTERMIDATE_POSITIONEMBEDK,
                                            INTERMIDATE_MIXEDV,
                                            IN_PASTKEY,
                                            IN_PASTVALUE,
                                            IN_ATTENTIONMASK,
                                            IN_TOKENOFFSET,
                                            IN_SEQLEN,
                                            IN_LAYERID};
    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionKvCacheNode.inTensorReshapeFuncs.resize(selfAttentionKvCacheNode.inTensorIds.size());
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };

    atb_speed::common::ParallelParamV2 selfOutLinearParam;
    selfOutLinearParam.commParam.rank = param.rank;
    selfOutLinearParam.commParam.rankSize = param.rankSize;
    selfOutLinearParam.commParam.backend = param.backend;
    selfOutLinearParam.isBias = false;
    atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {
        INTERMIDATE_SELFOUT,IN_SELFOUTLINEARWEIGHT, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    CreateOperation(rmsNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;
    mlpParam.commDownParam.backend = param.backend;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpParam.transposeB = true;
    mlpParam.isBias = false;
    mlpParam.isPack = false;
    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {
        INTERMIDATE_SELFNORMOUT,
        IN_MLPUPWEIGHT,
        IN_MLPGATEWEIGHT,
        IN_MLPDOWNWEIGHT,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
    };
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    CreateOperation(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LAYEROUT};

    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

FlashAttentionRopeLayerBinder::FlashAttentionRopeLayerBinder() = default;

FlashAttentionRopeLayerBinder::~FlashAttentionRopeLayerBinder() = default;

void FlashAttentionRopeLayerBinder::ParseParam(const nlohmann::json &paramJson)
{
    tokenOffset_.clear();
    for (const auto &item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }

    seqLen_.clear();
    for (const auto &item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
}

void FlashAttentionRopeLayerBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_TOKENOFFSET).hostData = tokenOffset_.data();
    variantPack.inTensors.at(IN_SEQLEN).hostData = seqLen_.data();
}
} // namespace baichuan2_7b
} // namespace atb_speedexport ASCEND_GLOBAL_LOG_LEVEL=3
' > ${ATB_FILE_NAME}.cpp

echo ""
echo ">>>> Transform quant"
echo -e "\033[1;32m[1/2]\033[0m msit_llm_transform about quant测试用例"
msit llm transform -s .
if [ $? -eq 0 ]
then
    echo msit_llm_transform about quant: Success
else
    echo msit_llm_transform about quant: Failed
    run_ok=$ret_failed
fi

MESSAGE=""
QUANT_ATB_FILE_NAME=quant_${ATB_FILE_NAME}
if [ ! -e ${QUANT_ATB_FILE_NAME}.cpp ]; then
    MESSAGE="$MESSAGE ${QUANT_ATB_FILE_NAME}.cpp"
    run_ok=$ret_failed
fi
if [ ! -e ${QUANT_ATB_FILE_NAME}.h ]; then
    MESSAGE="$MESSAGE ${QUANT_ATB_FILE_NAME}.h"
    run_ok=$ret_failed
fi

DEQSCALE_NUM=`grep -c _deqscale ${QUANT_ATB_FILE_NAME}.h`
if [ "$DEQSCALE_NUM" != "5" ]; then
    MESSAGE="$MESSAGE _deqscale"
    run_ok=$ret_failed
fi
BIAS_NUM=`grep -c _bias ${QUANT_ATB_FILE_NAME}.h`
if [ "$BIAS_NUM" != "5" ]; then
    MESSAGE="$MESSAGE _bias"
    run_ok=$ret_failed
fi

echo ""
echo ">>>> Transform sparse quant"
echo -e "\033[1;32m[2/2]\033[0m msit_llm_transform about sparse quant测试用例"
msit llm transform -s . --enable-sparse
if [ $? -eq 0 ]
then
    echo msit_llm_transform about sparse quant: Success
else
    echo msit_llm_transform about sparse quant: Failed
    run_ok=$ret_failed
fi

SPARSE_QUANT_ATB_FILE_NAME=sparse_quant_${ATB_FILE_NAME}
if [ ! -e ${SPARSE_QUANT_ATB_FILE_NAME}.cpp ]; then
    MESSAGE="$MESSAGE ${SPARSE_QUANT_ATB_FILE_NAME}.cpp"
    run_ok=$ret_failed
fi
if [ ! -e ${SPARSE_QUANT_ATB_FILE_NAME}.h ]; then
    MESSAGE="$MESSAGE ${SPARSE_QUANT_ATB_FILE_NAME}.h"
    run_ok=$ret_failed
fi

SPARSE_PARAM=`grep -c LinearSparseParam ${SPARSE_QUANT_ATB_FILE_NAME}.cpp`
if [ "$SPARSE_PARAM" != "1" ]; then
    MESSAGE="$MESSAGE LinearSparseParam"
    run_ok=$ret_failed
fi

echo ""
echo ">>>> Clean and revert"
cd $CUR_PATH
pip uninstall -y torch torch_npu apex
pip install torch==2.1.0
rm $TEST_DIR -rf

if [ $run_ok = "1" ]; then
    echo ">>>> [msit_llm_transform FAILED] $MESSAGE not exists"
    exit 1
else
    echo ">>>> Done!"
    exit 0
fi