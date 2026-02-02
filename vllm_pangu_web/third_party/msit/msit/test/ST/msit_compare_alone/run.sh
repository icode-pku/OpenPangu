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

echo -e "\033[1;32m[1/4]\033[0m msit_compare_alone 用例"
msit debug compare -mp $MODEL_PATH/alone_compare/onnx_om/npu/20240909202355/0/csc/1/0 -gp $MODEL_PATH/alone_compare/onnx_om/onnx --ops-json $MODEL_PATH/alone_compare/onnx_om/csc.json -o $OUTPUT_PATH
if [ $? -eq 0 ]
then
    echo msIT Compare onnx and om in the ${MODEL_PATH}: Success
else
    echo msIT Compare onnx and om in the ${MODEL_PATH}: Failed
    run_ok=$ret_failed
fi

echo -e "\033[1;32m[2/4]\033[0m msit_compare_alone 用例"
msit debug compare -mp $MODEL_PATH/alone_compare/saved_model_om/npu/20240815172310/0/ge_default_20240815172347_11/2/0 -gp $MODEL_PATH/alone_compare/saved_model_om/tf --ops-json $MODEL_PATH/alone_compare/saved_model_om/ge_proto_00000001_graph_11_Build.json -o $OUTPUT_PATH
if [ $? -eq 0 ]
then
    echo msIT Compare saved_model and om in the ${MODEL_PATH}: Success
else
    echo msIT Compare saved_model and om in the ${MODEL_PATH}: Failed
    run_ok=$ret_failed
fi

echo -e "\033[1;32m[3/4]\033[0m msit_compare_alone 用例"
msit debug compare -mp $MODEL_PATH/alone_compare/saved_model_cpu_npu/npu/20240905174530/0/ge_default_20240905174606_11/2/0 -gp $MODEL_PATH/alone_compare/saved_model_cpu_npu/tf --ops-json $MODEL_PATH/alone_compare/saved_model_cpu_npu/ge_proto_00000001_graph_11_Build.json -o $OUTPUT_PATH
if [ $? -eq 0 ]
then
    echo msIT Compare saved_model in cpu and npu in the ${MODEL_PATH}: Success
else
    echo msIT Compare saved_model in cpu and npu in the ${MODEL_PATH}: Failed
    run_ok=$ret_failed
fi

echo -e "\033[1;32m[4/4]\033[0m msit_compare_alone 用例"
msit debug compare -mp $MODEL_PATH/alone_compare/pb_om/npu/20240727172100/0/resnet50/1/0 -gp $MODEL_PATH/alone_compare/pb_om/tf --ops-json $MODEL_PATH/alone_compare/pb_om/resnet50.json -o $OUTPUT_PATH
if [ $? -eq 0 ]
then
    echo msIT Compare pb and om in the ${MODEL_PATH}: Success
else
    echo msIT Compare pb and om in the ${MODEL_PATH}: Failed
    run_ok=$ret_failed
fi

exit $run_ok