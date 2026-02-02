#!/bin/bash
# This script is used to run ut and st testcase.
# Copyright Huawei Technologies Co., Ltd. 2021-2025. All rights reserved.
CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=$(readlink -f ${CUR_DIR}/..)
TEST_DIR=${TOP_DIR}/"test"
SRC_DIR=${TOP_DIR}/"src"
ret=0

clean() {
  cd ${TEST_DIR}
  if [ -e ${TEST_DIR}/coverage.xml ]; then
    rm coverage.xml
    echo "remove last coverage.xml success"
  fi
  cd -
}

run_test_cpp() {
  cd .
}

run_test_python() {
  python3 --version
  
  export PYTHONPATH=${TOP_DIR}:${PYTHONPATH}
  
  python3 -m coverage run --branch --source ${TOP_DIR}/'msserviceprofiler' -m pytest ${TEST_DIR}/ut

  if [ $? -ne 0 ]; then
    echo "UT Failure"
    exit 1
  fi

  python3 -m coverage report -m
  python3 -m coverage xml -o ${TEST_DIR}/coverage.xml

  target_percentage=77
  limit_start_date="2025/9/5"
  percentage_str=`python3 -m coverage report -m | tail -1 | grep -oE '[0-9]+%' | tail -1`
  percentage=${percentage_str%\%}

  if [ "$percentage" -lt $target_percentage ]; then
    echo "====== 百分比 $percentage_str 小于 77% ======"
    target_timestamp=$(date -d "$limit_start_date" +%s)
    current_timestamp=$(date +%s)

    if [ "$current_timestamp" -gt "$target_timestamp" ]; then
      exit 1
    else
      echo "当前时间不大于 $limit_start_date, 尚不开启限制"
    fi
  else
    echo ""====== 百分比 $percentage_str 不小于 $target_percentage% ======""
  fi
}

run_test() {
  run_test_cpp
  run_test_python
}

main() {
  cd ${TEST_DIR}
  clean
  run_test
  echo "UT Success"
}

main