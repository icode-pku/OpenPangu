# Concatenate Command


## 介绍
对两幅ONNX图按照给定的输入输出映射关系进行拼接。

注意：若检测到待拼接的两幅ONNX图中存在命名冲突情况，将自动为第一幅ONNX图所有组件名称添加"pre_"前缀。

```bash
msit debug surgeon concat [OPTIONS]
```

concatenate 可简写为concat

参数说明：

| 参数                    | 说明                                                                                                                                                                  | 是否必选 |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| OPTIONS               | 额外参数。可取值：<br/>    -cgp/--combined-graph-path：拼接之后结构图的名称。默认为以下划线连接的两幅图的名称<br/>  -pref/--prefix:添加到第一幅ONNX图的前缀字符串，默认为"pre_"  <br/> -h, --help：工具使用帮助信息。                  | 否    |
| REQUIRED              | -g1/--graph1：输入的第一个ONNX模型，必须为.onnx文件。 <br/>    -g2/--graph2：输入的第一个ONNX模型，必须为.onnx文件。<br/>    -io/--io-map：拼接时第一幅图的输出与第二幅图的输入的映射关系。例如“g1_out1,g2_in1;g1_out2,g2_in2” | 是    |


## 运行示例

```bash
msit debug surgeon concat -g1 sub1.onnx -g2 sub2.onnx -io "g1_out1,g2_in1;g1_out2,g2_in2" 
```

输出示例如下：

```bash
2023-07-19 10:45:31,237 - 984068 - msit_debug_logger - WARNING - Cant merge two graphs with overlapping names. Found repeated nodes names: conv4_10/x1/bn_1_QuantizeLinear,Add
_nc_rename_45_quant,conv4_24/x1/bn_1_QuantizeLinear,Add_nc_rename_257_quant,concat_4_7_1_DequantizeLinear,Add_nc_rename_415_quant,BatchNormalization_nc_rename_453,Add_nc
_rename_67_quant...
2023-07-19 10:45:31,240 - 984068 - msit_debug_logger - INFO - A prefix `pre_` will be added to graph1
2023-07-19 10:45:31,510 - 984068 - msit_debug_logger - INFO - Concatenate ONNX model: densenet-12-int8.onnx and ONNX model: densenet-12-int8.onnx completed. Combined model sa
ved in res.onnx
```
