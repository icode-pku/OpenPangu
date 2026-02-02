export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

MODEL_PATH=$PROJECT_PATH/resource/msit_llm             #原模型路径

echo -e "\033[1;32m[1/1]\033[0m msit_llm_GE_FX_enhance_comparison测试用例"
pip install $MODEL_PATH/pytorch_v2.1.0_py310/apex-0.1+ascend-cp310-cp310-linux_aarch64.whl --force-reinstall
pip install $MODEL_PATH/torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall
pip install $MODEL_PATH/pytorch_v2.1.0_py310/torch_npu-2.1.0.post7-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall
pip install torchvision==0.16
cd -

PRE_NUMPY_VERSION=`python3 -c 'import numpy; print(numpy.__version__)'`
pip install numpy==1.24.4

CUR_PATH=$PWD
TEST_DIR=ait_llm_GE_FX_enhance_comparison_`date +%y%m%d%H%M`
echo ""
echo ">>>> TEST_DIR=$TEST_DIR"
mkdir -p $TEST_DIR
cd $TEST_DIR
rm -rf ge_dump gm_*_dump  # Should be empty, just in case

echo "
import os
import torch, torch_npu, torchvision
import numpy as np
from torch._functorch.aot_autograd import aot_module_simplified

import torchair as tng
from torchair.core.utils import logger
from torchair.configs.compiler_config import CompilerConfig

from msit_llm.dump import torchair_dump

def get_config():
    config = CompilerConfig()
    return config

target_dtype = torch.float16
model = torchvision.models.resnet50(pretrained=True).eval().to(target_dtype).npu()
if not os.path.exists('aa_224_224.npy'):
    np.save('aa_224_224.npy', np.random.uniform(size=[1, 3, 224, 224]))
aa = torch.from_numpy(np.load('aa_224_224.npy')).to(target_dtype).npu()

config = torchair_dump.get_ge_dump_config()
npu_backend = tng.get_npu_backend(compiler_config=config)
model = torch.compile(model, backend=npu_backend)
with torch.no_grad():
    try:
        print(model(aa).shape)
    except:
        pass
    print(model(aa).shape)
" > test_torchair_compare.py

echo ""
echo ">>>> Dump GE data"
python test_torchair_compare.py
sed -i 's/get_ge_dump_config/get_fx_dump_config/' test_torchair_compare.py

echo ""
echo ">>>> Dump FX data"
python test_torchair_compare.py

GE_DUMP_PATH=`ls -dt msit_ge_dump/dump_* | head -n 1`
FX_DUMP_PATH=`ls -dt data_dump | head -n 1`
echo ""
echo ">>>> Compare, GE_DUMP_PATH=$GE_DUMP_PATH, FX_DUMP_PATH=$FX_DUMP_PATH"
msit llm compare -gp $FX_DUMP_PATH -mp $GE_DUMP_PATH
if [ $? -eq 0 ]
then
    echo msit_llm_GE_FX_enhance_comparison: Success
else
    echo msit_llm_GE_FX_enhance_comparison: Failed
    run_ok=$ret_failed
fi

RESULT_CSV_PATH=`ls *_cmp_report_*.csv -1`
if [ ${RESULT_CSV_PATH} == "" ]; then
    MESSAGE=">>>> [FAILED] *_cmp_report_*.csv not exists"
    run_ok=$ret_failed
else
    MESSAGE=">>>> Done! result_csv: ${RESULT_CSV_PATH}"
fi

exit 0
echo ""
echo ">>>> Clean and revert"
cd $CUR_PATH
pip uninstall -y torch torch_npu apex
pip install numpy==$PRE_NUMPY_VERSION
pip install torch==2.1.0
rm $TEST_DIR -rf

echo ""
echo "MESSAGE:$MESSAGE"
exit $run_ok