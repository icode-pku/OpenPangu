export LD_PRELOAD=/home/ptq-test/anaconda3/envs/hwb_msit_smoke_py3.10/lib/python3.10/site-packages/torch.libs/libgomp-6e1a1d1b.so.1.0.0:$LD_PRELOAD
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

PROJECT_PATH=$(echo $PROJECT_PATH)                    #工程路径
TORCH_PATH=$PROJECT_PATH/resource/msit_llm        #原模型路径
MODEL_PATH=$PROJECT_PATH/resource/msit_compare        #原模型路径
OUTPUT_PATH=$PROJECT_PATH/output/msit_compare       #输出路径

pip install $TORCH_PATH/pytorch_v2.1.0_py310/apex-0.1+ascend-cp310-cp310-linux_aarch64.whl --force
pip install $TORCH_PATH/torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force
pip install $TORCH_PATH/pytorch_v2.1.0_py310/torch_npu-2.1.0.post7-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force

rm -rf $OUTPUT_PATH/*

echo -e "\033[1;32m[1/1]\033[0m Test case1 - msit_compare_mindie用例"
msit debug compare -gp $MODEL_PATH/mindie_cpu -mp $MODEL_PATH/mindie_npu --ops-json $MODEL_PATH/mapping_file  -o $OUTPUT_PATH
if [ $? -eq 0 ]
then
    echo msit compare about mindie in the ${MODEL_PATH}: Success
else
    echo msit compare about mindie in the ${MODEL_PATH}: Failed
    run_ok=$ret_failed
fi

echo "y" | pip uninstall torch --quiet
echo "y" | pip uninstall torch_npu --quiet
pip install torch==2.1.0

exit $run_ok