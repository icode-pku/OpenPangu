export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

PROJECT_PATH=$(echo $PROJECT_PATH)                    #工程路径

MODEL_PATH=$PROJECT_PATH/resource/msit_compare/        #原模型路径
OUTPUT_PATH=$PROJECT_PATH/output/msit_compare/       #输出路径

ASCEND_HOME_PATH=$(echo $ASCEND_HOME_PATH)

rm -rf $OUTPUT_PATH/*

main() {
    echo "[COMPARE INFO] aipp: RGB"
    cmd="msit debug compare -gm $MODEL_PATH/resnet18.onnx -om $MODEL_PATH/resnet18_bs8_aipp_no_fusion.om -is "image:8,3,224,224" -c $ASCEND_HOME_PATH -o $OUTPUT_PATH" #aipp
    $cmd || { echo "[COMPARE ERROR] msit debug compare case failed, cmd=$cmd";run_ok=$ret_failed; }
    echo "[COMPARE INFO] aipp: YUV420"
    cmd="msit debug compare -gm $MODEL_PATH/resnet18.onnx -om $MODEL_PATH/resnet18_bs8_aipp_no_fusion_YUV420.om -is "image:8,3,224,224" -c $ASCEND_HOME_PATH -o $OUTPUT_PATH"
    $cmd || { echo "[COMPARE ERROR] msit debug compare case failed, cmd=$cmd";run_ok=$ret_failed; }
    echo "[COMPARE INFO] aipp: YUV400"
    cmd="msit debug compare -gm $MODEL_PATH/resnet18.onnx -om $MODEL_PATH/resnet18_bs8_aipp_no_fusion_YUV400.om -is "image:8,3,224,224" -c $ASCEND_HOME_PATH -o $OUTPUT_PATH"
    $cmd || { echo "[COMPARE ERROR] msit debug compare case failed, cmd=$cmd";run_ok=$ret_failed; }
    echo "[COMPARE INFO] aipp: XRGB"
    cmd="msit debug compare -gm $MODEL_PATH/resnet18.onnx -om $MODEL_PATH/resnet18_bs8_aipp_no_fusion_XRGB.om -is "image:8,3,224,224" -c $ASCEND_HOME_PATH -o $OUTPUT_PATH"
    $cmd || { echo "[COMPARE ERROR] msit debug compare case failed, cmd=$cmd";run_ok=$ret_failed; }
    return $run_ok
}

main "$@"