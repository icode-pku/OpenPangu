# 单算子比对

## 介绍
精度比对工具单算子比对模式，支持对模型进行单算子并行比对，避免前置算子对后续算子的输入输出影响。
默认关闭，开启使用方式：`-single True` 或 `--single-op True`

## 运行示例

**前提**：暂不支持动态shape模型，运行过程中可能出现无法加载npy文件的情况，请添加`--onnx-fusion-switch False`以解决。
**注意**：
- 使用时请勿设置`--dump 为False`，默认为True.
- 不要开启`--custom-op`
- 不要开启`--locat`
执行命令如下：
```
msit debug compare -gm {onnx_model_path} -om {om_model_path} -i {input_data_path} -o {output_file_path} -single True
```

## 结果
输出文件在输出路径下的single_op文件夹中，比对结果汇总在`single_op_summary.csv`文件中。
比对结果说明请参见[对比结果分析步骤](../result_analyse/README.md)