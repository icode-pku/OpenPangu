export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

PROJECT_PATH=$(echo $PROJECT_PATH)                    #工程路径

MODEL_PATH=$PROJECT_PATH/resource/msit_compare        #原模型路径
OUTPUT_PATH=$PROJECT_PATH/output/msit_compare       #输出路径

rm -rf $OUTPUT_PATH/*

echo -e "\033[1;32m[1/1]\033[0m msit_compare_specify_input_shape_info用例"
msit debug compare -gm $MODEL_PATH/yolov3.onnx -om $MODEL_PATH/yolov3_bs2_new_linux_aarch64.om -dr "images:2,3,638-640,640" -o $OUTPUT_PATH #动态range shape
if [ $? -eq 0 ]
then
    echo msit compare about specify_input_shape_info in the ${MODEL_PATH}: Success
else
    echo msit compare about specify_input_shape_info in the ${MODEL_PATH}: Failed
    run_ok=$ret_failed
fi

exit $run_ok