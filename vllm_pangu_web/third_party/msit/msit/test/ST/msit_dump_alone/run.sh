export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok
PROJECT_PATH=$(echo $PROJECT_PATH)                 #工程路径
MODEL_PATH=$PROJECT_PATH/resource/msit_compare      #原模型路径
OUTPUT_PATH=$PROJECT_PATH/output/msit_compare       #输出路径

rm -rf $OUTPUT_PATH/*

source /home/ptq-test/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate binzai_py37

source $MODEL_PATH/ascend-toolkit/tfplugin/set_env.sh

bash $PROJECT_PATH/msit_update/run.sh    #检测是否需要更新msit

# 添加执行权限
chmod u+x ${PROJECT_PATH}/resource/msit_code/msit/msit/install.sh

# 安装msit
bash ${PROJECT_PATH}/resource/msit_code/msit/msit/install.sh --compare

echo -e "\033[1;32m[1/3]\033[0m Test case1 - msit_dump_alone用例"
msit debug dump -m $MODEL_PATH/csc.onnx -dp cpu -is 'input_ids:1,128;attention_mask:1,128;token_type_ids:1,128' -o $OUTPUT_PATH
if [ $? -eq 0 ]
then
    echo msIT Dump onnx models with cpu in the ${MODEL_PATH}: Success
else
    echo msIT Dump onnx models with cpu in the ${MODEL_PATH}: Failed
    run_ok=$ret_failed
fi

echo -e "\033[1;32m[2/3]\033[0m Test case2 - msit_dump_alone用例"
msit debug dump -m $MODEL_PATH/saved_model/model/resnet50 -dp npu --saved_model_tag_set serve -is 'input_1:1,224,224,3' -o $OUTPUT_PATH
if [ $? -eq 0 ]
then
    echo msIT Dump saved_model models with npu in the ${MODEL_PATH}: Success
else
    echo msIT Dump saved_model models with npu in the ${MODEL_PATH}: Failed
    run_ok=$ret_failed
fi

echo -e "\033[1;32m[3/3]\033[0m Test case2 - msit_dump_alone用例"
msit debug dump -m $MODEL_PATH/saved_model/model/resnet50 -dp cpu --saved_model_tag_set serve -is 'input_1:1,224,224,3' --tf-json $MODEL_PATH/ge_proto_00000001_graph_11_Build.json -o $OUTPUT_PATH
if [ $? -eq 0 ]
then
    echo msIT Dump saved_model models with cpu in the ${MODEL_PATH}: Success
else
    echo msIT Dump saved_model models with cpu in the ${MODEL_PATH}: Failed
    run_ok=$ret_failed
fi

conda activate hwb_msit_smoke_py3.10

exit $run_ok