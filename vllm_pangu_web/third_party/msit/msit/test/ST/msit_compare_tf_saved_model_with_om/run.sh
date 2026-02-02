source /home/ptq-test/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate binzai_py37

export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

PROJECT_PATH=$(echo $PROJECT_PATH)                    #工程路径
MODEL_PATH=$PROJECT_PATH/resource/msit_compare      #原模型路径
OUTPUT_PATH=$PROJECT_PATH/output/msit_compare       #输出路径

rm -rf $OUTPUT_PATH/*

source $MODEL_PATH/ascend-toolkit/tfplugin/set_env.sh

pip install $MODEL_PATH/saved_model/whl/ais_bench-0.0.2-py3-none-any.whl --force

bash $PROJECT_PATH/msit_update/run.sh    #检测是否需要更新msit

# 添加执行权限
chmod u+x ${PROJECT_PATH}/resource/msit_code/msit/msit/install.sh

# 安装msit
bash ${PROJECT_PATH}/resource/msit_code/msit/msit/install.sh --llm
bash ${PROJECT_PATH}/resource/msit_code/msit/msit/install.sh --compare

rm -rf $MODEL_PATH/saved_model/model/resnet50.pb
rm -rf $MODEL_PATH/saved_model/model/resnet50.om

# saved_model 转 pb
python run_resnet50.py

# pb 转 om
atc --model $MODEL_PATH/atc --model $MODEL_PATH/saved_model/model/resnet50.pb --output $MODEL_PATH/saved_model/model/resnet50 --soc_version Ascend310P3 --framework 3 --input_shape="input_1:1,224,224,3"

echo -e "\033[1;32m[1/2]\033[0m Test case1 - msit_compare_tf_saved_model_with_om单输入用例"
msit debug compare -gm $MODEL_PATH/saved_model/model/resnet50 -om $MODEL_PATH/saved_model/model/resnet50.om --saved_model_tag_set serve -is 'input_1:1,224,224,3' -o $OUTPUT_PATH --advisor -d 0
if [ $? -eq 0 ]
then
    echo msIT Compare singles inputs saved_model models with om models in the ${MODEL_PATH}: Success
else
    echo msIT Compare singles inputs saved_model models with om models in the ${MODEL_PATH}: Failed
    run_ok=$ret_failed
fi

rm -rf $MODEL_PATH/saved_model/model/conv2D.pb
rm -rf $MODEL_PATH/saved_model/model/conv2D.om

# saved_model 转 pb
python run_conv2D.py

# pb 转 om
atc --model $MODEL_PATH/saved_model/model/conv2D.pb --output $MODEL_PATH/saved_model/model/conv2D --soc_version Ascend310P3 --framework 3 --input_shape="input_1:16,32,32,3;input_2:1,16,16,32"

echo -e "\033[1;32m[2/2]\033[0m Test case2 - msit_compare_tf_saved_model_with_om多输入用例"
msit debug compare -gm $MODEL_PATH/saved_model/model/conv2D -om $MODEL_PATH/saved_model/model/conv2D.om --saved_model_tag_set serve -is 'input_1:16,32,32,3;input_2:1,16,16,32' -o $OUTPUT_PATH --advisor -d 0
if [ $? -eq 0 ]
then
    echo msIT Compare multiples inputs saved_model models with om models in the ${MODEL_PATH}: Success
else
    echo msIT Compare multiples inputs saved_model models with om models in the ${MODEL_PATH}: Failed
    run_ok=$ret_failed
fi

conda activate hwb_msit_smoke_py3.10

exit $run_ok