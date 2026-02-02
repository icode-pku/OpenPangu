# 关闭融合规则比较

## 介绍
精度比对工具关闭融合规则比对模式：一般情况下离线模型精度比对在转换离线模型时都是默认开启算子融合功能的。为了排查融合后的算子产生的精度问题，可以将模型转换时的算子融合功能关闭。

(1)对于om模型，生成关融合的dump数据文件，与开融合的dump数据文件进行比对；\
(2)对于TensorFlow框架，生成关融合的dump数据文件，和标杆模型进行比较。



默认关闭，开启使用方式：`--fusion-switch-file ./fusion_switch.cfg`，其中[fusion_switch.cfg]为昇腾融合规则配置文件，配置方法参见：[关闭融合规则](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/appdevg/aclcppdevg/aclcppdevg_000105.html)

**注意**：
- 使用时请勿关闭`--dump`，不要开启`--custom-op`、`--locat`、`--single-op`等高级功能，[-gm，--golden-model]入参必须为.onnx模型

## 运行示例
以关闭所有融合规则为例，[fusion_switch.cfg]配置方法如下：
```
{
    "Switch":{
        "GraphFusion":{
            "ALL":"off"
        },
        "UBFusion":{
            "ALL":"off"
         }
    }
}
```
将[fusion_switch.cfg]放在当前目录下，执行命令如下：
```
msit debug compare -gm {onnx_model_path} -om {om_model_path} -o {output_file_path} --fusion-switch-file ./fusion_switch.cfg
```

## 结果

- Tensor比对结果result_*.csv文件路径，参考[对比结果分析步骤](../result_analyse/README.md)。

- csv表格中输出参数和onnx比对om模型有区别，其中[GroundTruth]表示关闭算子融合的om离线模型算子名：

  | 参数               | 说明                        |
  |-----------------------| ----------------------- |
  | NPUDump   | 表示开启算子融合功能的离线模型的算子名。| 
  | GroundTruth| 表示关闭算子融合功能的离线模型的算子名。| 
