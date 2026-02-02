export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SLOG_PRINT_TO_STDOUT=0
declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

echo -e "\033[1;32m[1/1]\033[0m msit llm metrics 测试用例"

python3 scripts.py

if [ $? -eq 0 ]
then
    echo msit llm metrics: Success
    rm -f *.csv
else
    echo msit llm metrics: Failed
    run_ok=$ret_failed
fi

exit $run_ok