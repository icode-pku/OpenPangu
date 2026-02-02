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

echo -e "\033[1;32m[1/1]\033[0m Test case1 - msit_compare_retinanet_dym"
msit debug compare -gm $MODEL_PATH/retinanet.onnx -om $MODEL_PATH/retinanet.om -o $OUTPUT_PATH --advsion --input-shape="input0:1,3,1344,1344" --custom-op="BatchMultiClassNMS" -d 0
if [ $? -eq 0 ]
then
    echo msit_compare_retinanet_dym : Success
else
    echo msit_compare_retinanet_dym : Failed
    run_ok=$ret_failed
fi

exit $run_ok