# Alone Compare


## 介绍
用户也可以不通过参数传入模型，直接传入dump数据路径进行精度比对。


## 运行示例
- 命令示例 
- **注意：** 传入的-mp路径以及-gp路径必须为数据的上一层文件夹
  ```sh
  msit debug compare -mp /home/HwHiAiUser/onnx_prouce_data/resnet_offical_om -gp /home/HwHiAiUser/onnx_prouce_data/model/resnet_offical_onnx
  --ops-json /home/HwHiAiUser/onnx_prouce_data/resnet_offical/ops.json -o /home/HwHiAiUser/result/test
  ```
  - `-mp, --my-path` 指定npu侧dump数据路径 
  - `-gp, --golden-path` 指定cpu侧dump数据路径
  - `--ops-json ` 指定cpu侧与npu侧算子的匹配规则json文件路径
  - `-o, –-output` (可选) 输出文件路径，默认为当前路径
- cpu、npu侧dump数据的获取方式，参考[msit debug dump功能](../../../../../docs/debug/dump/README.md)

### 输出结果说明和分析步骤参考

请参考：[对比结果分析步骤](../result_analyse/README.md)