export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

MODEL_PATH=$PROJECT_PATH/resource/msit_llm             #原模型路径
OUTPUT_PATH=$PROJECT_PATH/output/msit_llm             #输出路径

rm -rf $OUTPUT_PATH/*

pip install $MODEL_PATH/pytorch_v2.1.0_py310/apex-0.1+ascend-cp310-cp310-linux_aarch64.whl --force-reinstall
pip install $MODEL_PATH/torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall
pip install $MODEL_PATH/pytorch_v2.1.0_py310/torch_npu-2.1.0.post7-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall

source $MODEL_PATH/atb/set_env.sh
source $MODEL_PATH/mindie_atb_models/set_env.sh

python solve_operator.py


echo -e "\033[1;32m[1/4]\033[0m msit_llm_opcheck测试用例"
msit llm opcheck -i $MODEL_PATH/tensors_5/ait_dump/tensors/0_1656694/0/ -o $OUTPUT_PATH -metric abs cos_sim kl
if [ $? -eq 0 ]
then
    echo msit_llm_opcheck: Success
else
    echo msit_llm_opcheck: Failed
    run_ok=$ret_failed
fi


echo -e "\033[1;32m[2/4]\033[0m msit_llm_opcheck支持多进程测试用例"
msit llm opcheck -i $MODEL_PATH/tensors_5/ait_dump/tensors/0_1656694/0/ -o $OUTPUT_PATH --jobs 2
if [ $? -eq 0 ]
then
    echo msit_llm_opcheck支持多进程: Success
else
    echo msit_llm_opcheck支持多进程: Failed
    run_ok=$ret_failed
fi


echo -e "\033[1;32m[3/4]\033[0m msit_llm_opcheck支持优化项识别测试用例"
msit llm opcheck -i $MODEL_PATH/tensors_5/ait_dump/tensors/0_1656694/0/ -o $OUTPUT_PATH -rerun -opt
if [ $? -eq 0 ]
then
    echo msit_llm_opcheck支持优化项识别: Success
else
    echo msit_llm_opcheck支持优化项识别: Failed
    run_ok=$ret_failed
fi


echo -e "\033[1;32m[4/4]\033[0m msit_llm_opcheck支持量化模型测试用例"
msit llm opcheck -i $MODEL_PATH/tensors_8/ait_dump_20240808_081400/tensors/0_1799853/0 -o $OUTPUT_PATH 
if [ $? -eq 0 ]
then
    echo msit_llm_opcheck支持量化模型: Success
else
    echo msit_llm_opcheck支持量化模型: Failed
    run_ok=$ret_failed
fi


echo "uninstall torch" | pip uninstall torch --quiet
echo "uninstall torch_npu" | pip uninstall torch_npu --quiet
pip install torch==2.1.0

exit $run_ok