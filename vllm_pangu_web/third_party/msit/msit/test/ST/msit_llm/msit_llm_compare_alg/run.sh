export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0

MODEL_NAME=ait_llm_compare_alg
CUR_PATH=$PWD
TEST_DIR=${MODEL_NAME}_`date +%y%m%d%H%M`

MODEL_PATH=$PROJECT_PATH/resource/msit_llm             #原模型路径

pip install $MODEL_PATH/pytorch_v2.1.0_py310/apex-0.1+ascend-cp310-cp310-linux_aarch64.whl --force-reinstall
pip install $MODEL_PATH/torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall
pip install $MODEL_PATH/pytorch_v2.1.0_py310/torch_npu-2.1.0.post7-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall

echo ""
echo ">>>> TEST_DIR=$TEST_DIR"
mkdir -p $TEST_DIR
cd $TEST_DIR

FAKE_PATH="msit_dump/tensors/0/foo"
mkdir -p $FAKE_PATH/after

python3 -c "
import os
import numpy as np

golden = np.array([1, 11, 7, 5, 5, 4, 7, 6], dtype='float16')
np.save('golden.npy', golden)

header = b'\$Version=1.0\n\$Object.Count=1\n\$Object.Length=8\nformat=2\ndtype=1\ndims=2,4\n\$Object.data=0,8\n\$End=1\n'
with open(os.path.join('$FAKE_PATH','after','intensor0.bin'), 'wb') as ff:
    ff.write(header+golden.tobytes())
"
echo "
def custom_acc(golden, my):
    return (golden - my).sum().item(), ''

def custom_acc_2(golden, my):
    return (golden - my).mean().item(), ''
" > test_alg.py

msit llm compare -gp golden.npy -mp $FAKE_PATH/after/intensor0.bin -alg test_alg.py:custom_acc test_alg.py:custom_acc_2 > foo.log 2>&1

sed -i "s/'/\"/g" foo.log
python3 -c '
import json
with open("foo.log") as ff:
    cc = ff.read().split("Compared results:")[-1].split("\n")[0].strip()
    aa = json.loads(cc)
assert "custom_acc" in aa and aa["custom_acc"] == 0.0
assert "custom_acc_2" in aa and aa["custom_acc_2"] == 0.0
'
RESULT=$?

cd $CUR_PATH
rm $TEST_DIR -rf

echo "uninstall torch" | pip uninstall torch --quiet
echo "uninstall torch_npu" | pip uninstall torch_npu --quiet
pip install torch==2.1.0

exit $RESULT