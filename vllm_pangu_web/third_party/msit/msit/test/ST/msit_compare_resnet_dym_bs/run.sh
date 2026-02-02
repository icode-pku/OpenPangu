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

echo -e "\033[1;32m[1/1]\033[0m Test case1 - msit_compare_resnet_dym_bs"
msit debug compare -gm $MODEL_PATH/resnet18.onnx -om $MODEL_PATH/resnet18_bs8_no_fusion.om -o $OUTPUT_PATH --advsion -is "image:8,3,224,224" -d 0
if [ $? -eq 0 ]
then
    echo msit_Compare_resnet_dym_bs : Success
else
    echo msit_Compare_resnet_dym_bs : Failed
    run_ok=$ret_failed
fi

exit $run_ok