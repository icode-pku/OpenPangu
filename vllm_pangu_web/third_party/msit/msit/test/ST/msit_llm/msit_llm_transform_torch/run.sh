export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0

MODEL_NAME=msit_llm_transform_torch #当前用例名称

MODEL_PATH=$PROJECT_PATH/resource/msit_llm             #原模型路径

echo 操作记录: $EXECUT_RESULT_FILE_PATH
echo -e "\033[1;32m[1/1]\033[0m ${MODEL_NAME}测试用例"

pip install $MODEL_PATH/pytorch_v2.1.0_py310/apex-0.1+ascend-cp310-cp310-linux_aarch64.whl --force-reinstall
pip install $MODEL_PATH/torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall
pip install $MODEL_PATH/pytorch_v2.1.0_py310/torch_npu-2.1.0.post7-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall

cd -

CUR_PATH=$PWD
TEST_DIR=${MODEL_NAME}_`date +%y%m%d%H%M`
ATB_FILE_NAME="llamaforcausallm"
LLAMA_PATH_NAME="test_llama"
echo ""
echo ">>>> TEST_DIR=$TEST_DIR"
mkdir -p $TEST_DIR
cd $TEST_DIR

mkdir -p $LLAMA_PATH_NAME
echo '
{
    "architectures": [
      "LlamaForCausalLM"
    ],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 512,
    "initializer_range": 0.02,
    "intermediate_size": 1376,
    "max_position_embeddings": 512,
    "model_type": "llama",
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "num_key_value_heads": 4,
    "pad_token_id": 0,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.31.0.dev0",
    "use_cache": true,
    "vocab_size": 4000
}
' > $LLAMA_PATH_NAME/config.json

echo ""
echo ">>>> Transform torch model to atb"
msit llm transform -s $LLAMA_PATH_NAME

RESULT=$?
MESSAGE=""
ATB_REAULT_FILE_NAME=${ATB_FILE_NAME}/model/decoder_model.cpp
if [ ! -e $ATB_REAULT_FILE_NAME ]; then
    RESULT=1
    MESSAGE="$MESSAGE ${ATB_REAULT_FILE_NAME}"
fi
ATB_REAULT_FILE_NAME=${ATB_FILE_NAME}/model/decoder_model.h
if [ ! -e $ATB_REAULT_FILE_NAME ]; then
    RESULT=1
    MESSAGE="$MESSAGE ${ATB_REAULT_FILE_NAME}"
fi
ATB_REAULT_FILE_NAME=${ATB_FILE_NAME}/layer/decoder_layer.cpp
if [ ! -e $ATB_REAULT_FILE_NAME ]; then
    RESULT=1
    MESSAGE="$MESSAGE ${ATB_REAULT_FILE_NAME}"
fi
ATB_REAULT_FILE_NAME=${ATB_FILE_NAME}/layer/decoder_layer.h
if [ ! -e $ATB_REAULT_FILE_NAME ]; then
    RESULT=1
    MESSAGE="$MESSAGE ${ATB_REAULT_FILE_NAME}"
fi

echo ""
echo ">>>> Clean and revert"
cd $CUR_PATH
rm $TEST_DIR -rf

echo "uninstall torch" | pip uninstall torch --quiet
echo "uninstall torch_npu" | pip uninstall torch_npu --quiet
pip install torch==2.1.0

if [ ${RESULT} = "1" ]; then
    echo ">>>> [FAILED] $MESSAGE not exists"
    exit 1
else
    echo ">>>> Done!"
    exit 0
fi