export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

PROJECT_PATH=$(echo $PROJECT_PATH)                    #工程路径

echo -e "\033[1;32m[1/1]\033[0m msit_compare_onnx_over_2G 用例"

function prepare_onnx() {
	python3 scripts.py
	if [ $? -ne 0 ]; then
		echo "Fatal: command 'python3 scripts.py' failed. Please check 'scripts.py'."

		return 1
	fi

	if [[ ! -d "temp_dir" ]]; then
		echo "Error: 'temp_dir' should have being created but not found. Please check the python script."

		return 1
	fi

	cd "temp_dir"

	if [[ ! -f "large_model.onnx" ]]; then
		echo "Error: 'large_model.onnx' should have being created but not found under 'temp_dir'. Please check the python script."

		return 1
	fi
}

function prepare_om() {
	if [ $? -ne 0 ]; then
		return 2
	fi

	echo "Info: onnx model is ready. Now preparing om model."
	atc --framework 5 --model large_model.onnx --soc_version Ascend310P3 --output large_model
}

function main() {
	if [ $? -ne 0 ]; then
		return 3
	fi

	msit debug compare -gm "large_model.onnx" -om "large_model.om"
}

prepare_onnx
prepare_om
main

if [ $? -eq 0 ]
then
    echo "msit debug compare in the onnx over 2G case: Success"
else
    echo "msit debug compare in the onnx over 2G case: Failed"

    run_ok=$ret_failed
fi

rm -rf "$(dirname "$0")/temp_dir"

exit $run_ok