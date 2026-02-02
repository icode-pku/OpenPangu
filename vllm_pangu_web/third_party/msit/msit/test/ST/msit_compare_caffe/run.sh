source /home/ptq-test/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate hwb_msit_smoke
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

PROJECT_PATH=$(echo $PROJECT_PATH)                    #工程路径

bash $PROJECT_PATH/msit_update/run.sh    #检测是否需要更新msit

# 添加执行权限
chmod u+x ${PROJECT_PATH}/resource/msit_code/msit/msit/install.sh

# 安装msit
bash ${PROJECT_PATH}/resource/msit_code/msit/msit/install.sh --compare

MODEL_PATH=$PROJECT_PATH/resource/msit_compare        #原模型路径
OUTPUT_PATH=$PROJECT_PATH/output/msit_compare       #输出路径

ASCEND_HOME_PATH=$(echo $ASCEND_HOME_PATH)

rm -rf $OUTPUT_PATH/*

echo -e "\033[1;32m[1/1]\033[0m msit_compare_caffe用例"
msit debug compare -gm $MODEL_PATH/resnet50.prototxt -w $MODEL_PATH/resnet50.caffemodel -om $MODEL_PATH/resnet50.om -c $ASCEND_HOME_PATH -o $OUTPUT_PATH #caffe
if [ $? -eq 0 ]
then
    echo msit compare caffe models in the ${MODEL_PATH}: Success
else
    echo msit compare caffe models in the ${MODEL_PATH}: Failed
    run_ok=$ret_failed
fi

conda activate hwb_msit_smoke_py3.10

exit $run_ok