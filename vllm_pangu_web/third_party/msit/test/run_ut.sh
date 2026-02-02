#!/bin/bash

CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=$(readlink -f ${CUR_DIR}/..)
TEST_DIR=${TOP_DIR}/"test"

clean() {
    if [ -e $1 ]; then
        rm -f $1
        echo "remove last $1 success"
    fi
}

run_test() {
    cd $1
    clean ./.coverage
    clean ./coverage.xml

    # 执行该目录底下的ut
    bash run_ut.sh
}

main() {
    clean ${TEST_DIR}/.coverage
    clean ${TEST_DIR}/coverage.xml

    # 执行各组件目录下的ut文件
    run_test $TOP_DIR/msit/test/UT
    run_test $TOP_DIR/msserviceprofiler/test
    run_test $TOP_DIR/msmodelslim/test
    echo "All ut success"

    # 使用combine生成统一的覆盖率报告
    cd ${TOP_DIR}
    coverage combine msit/test/UT/.coverage msserviceprofiler/test/.coverage msmodelslim/test/.coverage
    python3 -m coverage report -m
    python3 -m coverage xml -o ${TEST_DIR}/coverage.xml
    echo "Generate total coverage report success"

    cd ${CUR_DIR}
}

main