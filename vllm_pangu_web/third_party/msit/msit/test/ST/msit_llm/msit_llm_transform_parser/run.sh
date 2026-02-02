export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

MODEL_PATH=$PROJECT_PATH/resource/msit_llm/model_parser             #原模型路径
OUTPUT_PATH=$PROJECT_PATH/output/ait_llm

pip install $MODEL_PATH/transformers-4.41.1-py3-none-any.whl

echo "
from json import dump

from transformers import AutoModelForCausalLM, AutoConfig

from msit_llm.transform.model_parser.parser import build_model_tree

conf1 = AutoConfig.from_pretrained('$MODEL_PATH/llama2', trust_remote_code=True)
model1 = AutoModelForCausalLM.from_config(conf1, trust_remote_code=True)
with open('$OUTPUT_PATH/llama2.json', 'w') as o:
    dump(build_model_tree(model1), o)

conf2 = AutoConfig.from_pretrained('$MODEL_PATH/chatglm2', trust_remote_code=True)
model2 = AutoModelForCausalLM.from_config(conf2, trust_remote_code=True)
with open('$OUTPUT_PATH/chatglm2.json', 'w') as o:
    dump(build_model_tree(model2), o)

conf3 = AutoConfig.from_pretrained('$MODEL_PATH/baichuan2', trust_remote_code=True)
model3 = AutoModelForCausalLM.from_config(conf3, trust_remote_code=True)
with open('$OUTPUT_PATH/baichuan2.json', 'w') as o:
    dump(build_model_tree(model3), o)
" > test.py

python test.py
if [ $? -eq 0 ]
then
    echo msit_llm_transform_parser: Success
else
    echo msit_llm_transform_parser: Failed
    run_ok=$ret_failed
fi

pip install transformers==4.30.2 -q

exit $run_ok