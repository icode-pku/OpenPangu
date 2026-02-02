# ais_bench 推理工具使用指南
- 用来针对指定的推理模型运行推理程序，并能够测试推理模型的性能（包括吞吐率、时延）。
- **该部分代码已移至 [Gitee Ascend/tools/ais-bench_workload](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)，文档及安装方法可参照对应说明**
- **安装方式** 可使用 [Gitee Ascend/tools/ais-bench_workload](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench) 提供的 whl 包安装
  ```sh
  pip3 install aclruntime-{version}-{python_version}-linux_{arch}.whl
  pip3 install ais_bench-{version}-py3-none-any.whl
  ```
  分别提示如下信息则表示安装成功：
  ```sh
  # 成功安装aclruntime
  Successfully installed aclruntime-{version}
  # 成功安装ais_bench推理程序
  Successfully installed ais_bench-{version}
  ```