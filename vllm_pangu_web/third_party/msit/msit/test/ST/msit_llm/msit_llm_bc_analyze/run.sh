export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok


MODEL_PATH=$PROJECT_PATH/resource/msit_llm             #原模型路径

echo -e "\033[1;32m[1/1]\033[0m msit_llm_recognize_bad_case测试用例"

pip install $MODEL_PATH/pytorch_v2.1.0_py310/apex-0.1+ascend-cp310-cp310-linux_aarch64.whl --force-reinstall
pip install $MODEL_PATH/torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall
pip install $MODEL_PATH/pytorch_v2.1.0_py310/torch_npu-2.1.0.post7-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall

msit llm analyze -g $MODEL_PATH/msit_llm_bc_analyze/golden.csv -t $MODEL_PATH/msit_llm_bc_analyze/test.csv

if [$? -eq 0 ]
then
    echo msit_llm_bc_analyze: Success
else
    echo msit_llm_bc_analyze: Failed
    run_ok=$ret_failed
fi

echo "uninstall torch" | pip uninstall torch --quiet
echo "uninstall torch_npu" | pip uninstall torch_npu --quiet
pip install torch==2.1.0

exit $run_ok