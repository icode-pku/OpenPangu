export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok
PROJECT_PATH=$(echo $PROJECT_PATH)                 #工程路径
MODEL_PATH=$PROJECT_PATH/resource/msit_llm      #原模型路径
OUTPUT_PATH=$PROJECT_PATH/output/msit_compare       #输出路径

rm -rf $OUTPUT_PATH/*

pip install $MODEL_PATH/pytorch_v2.1.0_py310/apex-0.1+ascend-cp310-cp310-linux_aarch64.whl --force
pip install $MODEL_PATH/torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force
pip install $MODEL_PATH/pytorch_v2.1.0_py310/torch_npu-2.1.0.post7-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force

source $MODEL_PATH/mindie/set_env.sh

echo -e "\033[1;32m[1/1]\033[0m Test case1 - msit_dump_mindietorch用例"

msit debug dump --output $OUTPUT_PATH --exec "python test.py"
if [ $? -eq 0 ]
then
    echo msIT Dump mindie_torch inference model script: Success
else
    echo msIT Dump mindie_torch inference model script: Failed
    run_ok=$ret_failed
fi

echo "y" | pip uninstall torch --quiet
echo "y" | pip uninstall torch_npu --quiet
pip install torch==2.1.0

exit $run_ok