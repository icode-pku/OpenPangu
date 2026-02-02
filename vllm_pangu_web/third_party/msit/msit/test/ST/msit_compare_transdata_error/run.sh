export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

PROJECT_PATH=$(echo $PROJECT_PATH)                    #工程路径

MODEL_PATH=$PROJECT_PATH/resource/msit_compare        #原模型路径
OUTPUT_PATH=$PROJECT_PATH/output/msit_compare       #输出路径

ASCEND_HOME_PATH=$(echo $ASCEND_HOME_PATH)

rm -rf $OUTPUT_PATH/*

echo -e "\033[1;32m[1/1]\033[0m msit_compare_transdata_error 用例"
msit debug compare -gm $MODEL_PATH/Crane.onnx -om $MODEL_PATH/Crane.om \
-c $ASCEND_HOME_PATH -o $OUTPUT_PATH
if [ $? -eq 0 ]
then
    echo Msit Compare transdata in the ${MODEL_PATH}: Success
else
    echo Msit Compare transdata in the ${MODEL_PATH}: Failed
    run_ok=$ret_failed
fi

exit $run_ok