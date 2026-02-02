export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0


MODEL_PATH=$PROJECT_PATH/resource/msit_llm             #原模型路径

MODEL_NAME=msit_llm_opcheck_alg
CUR_PATH=$PWD
TEST_DIR=${MODEL_NAME}_`date +%y%m%d%H%M`

echo -e "\033[1;32m[1/1]\033[0m msit_llm_opcheck_alg测试用例"
pip install $MODEL_PATH/pytorch_v2.1.0_py310/apex-0.1+ascend-cp310-cp310-linux_aarch64.whl --force-reinstall
pip install $MODEL_PATH/torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall
pip install $MODEL_PATH/pytorch_v2.1.0_py310/torch_npu-2.1.0.post7-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall


echo ""
echo ">>>> TEST_DIR=$TEST_DIR"
mkdir -p $TEST_DIR
cd $TEST_DIR

FAKE_PATH="msit_dump/tensors/foo/0/0_LinearOperation"
mkdir -p $FAKE_PATH/after

python3 -c "
import os
import numpy as np

golden = np.array([1, 11, 7, 5, 5, 4, 7, 6] * 4, dtype='float16')

header = b'\$Version=1.0\n\$Object.Count=1\n\$Object.Length=32\nformat=2\ndtype=1\ndims=2,16\n\$Object.data=0,32\n\$End=1\n'
with open(os.path.join('$FAKE_PATH','after','intensor0.bin'), 'wb') as ff:
    ff.write(header+golden.tobytes())

with open(os.path.join('$FAKE_PATH','after','intensor1.bin'), 'wb') as ff:
    ff.write(header+golden[::-1].tobytes())

header = b'\$Version=1.0\n\$Object.Count=1\n\$Object.Length=4\nformat=2\ndtype=1\ndims=2,2\n\$Object.data=0,4\n\$End=1\n'
rr = golden.reshape([2, 16]) @ golden[::-1].reshape([2, 16]).T
with open(os.path.join('$FAKE_PATH','after','outtensor0.bin'), 'wb') as ff:
    ff.write(header+rr.tobytes())


pp = '{\"hasBias\":false,\"outDataType\":-1,\"transposeA\":false,\"transposeB\":true}'
with open(os.path.join('$FAKE_PATH','op_param.json'), 'w') as ff:
    ff.write(pp)
"

echo "
def custom_acc(golden, my):
    return (golden - my).sum().item(), ''

def custom_acc_2(golden, my):
    return (golden - my).mean().item(), ''
" > test_alg.py

msit llm opcheck -i $FAKE_PATH -alg test_alg.py:custom_acc test_alg.py:custom_acc_2

python3 -c '
import os
import glob
import pandas as pd

rr = sorted(glob.glob("opcheck_result*.xlsx"), key=lambda xx: os.path.getmtime(xx))[-1]
tt = pd.read_excel(rr)

assert "custom_acc" in tt.columns
assert "custom_acc_2" in tt.columns
'
RESULT=$?

cd $CUR_PATH
rm $TEST_DIR -rf

echo "uninstall torch" | pip uninstall torch --quiet
echo "uninstall torch_npu" | pip uninstall torch_npu --quiet
pip install torch==2.1.0

exit $RESULT