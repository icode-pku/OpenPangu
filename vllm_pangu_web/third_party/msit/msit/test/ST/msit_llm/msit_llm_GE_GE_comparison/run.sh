export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

MODEL_PATH=$PROJECT_PATH/resource/msit_llm             #原模型路径
FUSION_SWITCH_FILE="fusion_switch.json"

echo -e "\033[1;32m[1/1]\033[0m msit_llm_GE_GE_comparison测试用例"
pip install $MODEL_PATH/pytorch_v2.1.0_py310/apex-0.1+ascend-cp310-cp310-linux_aarch64.whl --force-reinstall
pip install $MODEL_PATH/torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall
pip install $MODEL_PATH/pytorch_v2.1.0_py310/torch_npu-2.1.0.post7-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall
cd -

PRE_NUMPY_VERSION=`python3 -c 'import numpy; print(numpy.__version__)'`
pip install numpy==1.24.4

CUR_PATH=$PWD
TEST_DIR=${MODEL_NAME}_`date +%y%m%d%H%M`
echo ""
echo ">>>> TEST_DIR=$TEST_DIR"
mkdir -p $TEST_DIR
cd $TEST_DIR

echo '{
    "Switch": {
      "GraphFusion": {
        "ALL": "off"
    },
    "UBFusion": {
      "ALL": "off"
    }
  }
}' > $FUSION_SWITCH_FILE

echo "
import os
import numpy as np
import torch, torch_npu
import torchair
from msit_llm.dump import torchair_dump

class LlamaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.var = 9.9

    def forward(self, x, y):
        add1 = x + self.var
        add2 = add1 + x
        mul1 = y.view([-1]) * 1.0
        mul2 = mul1 * add2
        view1 = mul2.view([-1,128])
        arange1 = torch.arange(view1.size(-1)).to(x.device)
        unsquese1 = torch.unsqueeze(arange1, 0)
        clone1 = unsquese1.clone()
        transpose1 = torch.transpose(clone1, 0, 1)
        div1 = view1 / 0.1
        mm1 = torch.matmul(div1, transpose1.to(torch.int).to(target_dtype))
        expand1 = mm1.expand([64, 128])
        full1 = torch.full((mm1.size(0), mm1.size(0)), 1.0).to(x.device)
        cat1 = torch.cat([full1, -full1], -1)
        expand1 = expand1 + cat1
        pow1 = expand1.pow(2)
        repeat = pow1.repeat([2,1,1])
        unsafe_view1 = repeat.reshape(repeat.size(0), repeat.size(2), repeat.size(1))
        indices = unsafe_view1[1:2, :32, :-1].to(torch.bool).to(torch.int64)
        gather1 = torch.gather(repeat, 1, indices)
        softmax1 = torch.nn.functional.softmax(gather1, dim=-1, dtype=target_dtype)
        rsqrt1 = torch.rsqrt(softmax1)
        rsub = 1.0 - rsqrt1
        silu = torch.nn.functional.silu(rsub)
        eb1 = torch.embedding(y, silu.view([32, -1]).to(torch.long), -1, False, False)
        lt = silu[:1, :1, :1] < eb1[:1, :1, :1]
        return lt

if not os.path.exists('aa_8192.npy'):
    np.save('aa_8192.npy', np.random.uniform(size=[8192]).astype('float32'))
if not os.path.exists('aa_64_128.npy'):
    np.save('aa_64_128.npy', np.random.uniform(size=[64, 128]).astype('float32'))

target_dtype = torch.float16
model = LlamaModel().to(target_dtype).npu()
aa = torch.from_numpy(np.load('aa_8192.npy')).to(target_dtype).npu()
bb = torch.from_numpy(np.load('aa_64_128.npy')).to(target_dtype).npu()

config = torchair_dump.get_ge_dump_config(dump_path='ge_dump')
npu_backend = torchair.get_npu_backend(compiler_config=config)
model = torch.compile(model, backend=npu_backend)

print(model(aa, bb).shape)
print(model(aa, bb).shape)
" > test_torchair_compare.py

echo ""
echo ">>>> Dump GE data"
python test_torchair_compare.py

echo ""
echo ">>>> Dump GE fusion off data"
sed -i "s/dump_path='ge_dump'/dump_path='fusion_off_ge_dump', fusion_switch_file='fusion_switch.json'/" test_torchair_compare.py
python test_torchair_compare.py

GE_DUMP_PATH=`ls -dt ge_dump/dump_* | head -n 1`
FUSION_OFF_GE_DUMP_PATH=`ls -dt fusion_off_ge_dump/dump_* | head -n 1`
echo ""
echo ">>>> Compare, GE_DUMP_PATH=$GE_DUMP_PATH, FUSION_OFF_GE_DUMP_PATH=$FUSION_OFF_GE_DUMP_PATH"
msit llm compare -gp $FUSION_OFF_GE_DUMP_PATH -mp $GE_DUMP_PATH
if [ $? -eq 0 ]
then
    echo msit_llm_GE_GE_comparison: Success
else
    echo msit_llm_GE_GE_comparison: Failed
    run_ok=$ret_failed
fi

RESULT_CSV_PATH=`ls *_cmp_report_*.csv -1`
if [ "${RESULT_CSV_PATH}" == "" ]; then
    MESSAGE=">>>> [FAILED] *_cmp_report_*.csv not exists"
    run_ok=$ret_failed
else
    MESSAGE=">>>> Done! result_csv: ${RESULT_CSV_PATH}"
fi

FUSION_LINES=`grep -c AutomaticBufferFusionOp $RESULT_CSV_PATH`
if [ "${FUSION_LINES}" != "18" ]; then
    MESSAGE="$MESSAGE, AutomaticBufferFusionOp=$FUSION_LINES count not right"
    run_ok=$ret_failed
fi

echo ""
echo ">>>> Clean and revert"
cd $CUR_PATH
pip uninstall -y torch torch_npu apex
pip install numpy==$PRE_NUMPY_VERSION
pip install torch==2.1.0 
rm $TEST_DIR -rf

echo ""
echo $MESSAGE
exit $run_ok