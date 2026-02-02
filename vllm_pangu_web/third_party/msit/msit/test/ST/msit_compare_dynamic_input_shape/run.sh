export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0

MODEL_NAME=run_compare_dynamic_input_shape
CUR_PATH=$PWD
TEST_DIR=${MODEL_NAME}_`date +%y%m%d%H%M`

echo ""
echo ">>>> TEST_DIR=$TEST_DIR"
mkdir -p $TEST_DIR
cd $TEST_DIR

MODEL_NAME="resnet18_dynamic"
python3 -c "
import torch, torchvision

mm = torchvision.models.resnet18()
dynamic_axes = {'input': {0: '-1', 2: '-1', 3: '-1'}}
torch.onnx.export(mm, torch.ones([1, 3, 224, 224]), '$MODEL_NAME.onnx', input_names=['input'], dynamic_axes=dynamic_axes)
"

chmod 640 ${MODEL_NAME}.onnx
atc --framework 5 --model ${MODEL_NAME}.onnx --soc_version Ascend310P3 --input_format NCHW \
--input_shape "input:1,3,32~64,32~64" --output ${MODEL_NAME}
msit debug compare -gm ${MODEL_NAME}.onnx -om ${MODEL_NAME}_linux_aarch64.om -is "input:1,3,64,64"
RESULT=$?

cd $CUR_PATH
rm $TEST_DIR -rf
exit $RESULT