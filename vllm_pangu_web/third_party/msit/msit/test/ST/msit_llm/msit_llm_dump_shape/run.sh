export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

MODEL_PATH=$PROJECT_PATH/resource/msit_llm             #原模型路径
OUTPUT_PATH=$PROJECT_PATH/output/msit_llm                #输出路径

rm -rf $OUTPUT_PATH/*

echo -e "\033[1;32m[1/1]\033[0m msit_llm_dump_Json2Onnx测试用例"
pip install $MODEL_PATH/pytorch_v2.1.0_py310/apex-0.1+ascend-cp310-cp310-linux_aarch64.whl --force-reinstall
pip install $MODEL_PATH/torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall
pip install $MODEL_PATH/pytorch_v2.1.0_py310/torch_npu-2.1.0.post7-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall

source $MODEL_PATH/atb/set_env.sh
source $MODEL_PATH/mindie_atb_models/set_env.sh

cd $MODEL_PATH/mindie_atb_models/examples
msit llm dump --exec "bash $MODEL_PATH/mindie_atb_models/examples/models/chatglm/v2_6b/run_300i_duo_pa.sh $MODEL_PATH/chatglm2_6b" -sd -o $OUTPUT_PATH --type onnx #冒烟用例脚本，需根据自己的用例需要修改
if [ $? -eq 0 ]
then
    echo msit_llm_dump_Json2Onnx: Success
else
    echo msit_llm_dump_Json2Onnx: Failed
    run_ok=$ret_failed
fi

echo "uninstall torch" | pip uninstall torch --quiet
echo "uninstall torch_npu" | pip uninstall torch_npu --quiet
pip install torch==2.1.0

exit $run_ok