# Dynamic AIPP
## 介绍
- 动态AIPP的介绍参考[什么是AIPP](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devaids/auxiliarydevtool/atlasatc_16_0016.html)。
- 目前benchmark工具只支持单个input的带有动态AIPP配置的模型，只支持静态shape、动态batch、动态宽高三种场景，不支持动态shape场景。
## 运行示例
### --aipp-config 输入的.config文件模板
以resnet18模型所对应的一种aipp具体配置为例(actual_aipp_conf.config)：
```cfg
[aipp_op]
    input_format : RGB888_U8
    src_image_size_w : 256
    src_image_size_h : 256

    crop : 1
    load_start_pos_h : 16
    load_start_pos_w : 16
    crop_size_w : 224
    crop_size_h : 224

    padding : 0
    csc_switch : 0
    rbuv_swap_switch : 0
    ax_swap_switch : 0
    csc_switch : 0

	  min_chn_0 : 123.675
	  min_chn_1 : 116.28
	  min_chn_2 : 103.53
	  var_reci_chn_0 : 0.0171247538316637
	  var_reci_chn_1 : 0.0175070028011204
	  var_reci_chn_2 : 0.0174291938997821
```
- .config文件`[aipp_op]`下的各字段名称及其取值范围参考CANN手册中的[静态AIPP配置实例](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devaids/auxiliarydevtool/atlasatc_16_0019.html)与[动态AIPP配置实例](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devaids/auxiliarydevtool/atlasatc_16_0020.html)。
- .config文件`[aipp_op]`下的`input_format`、`src_image_size_w`、`src_image_size_h`字段是必填字段。
- .config文件中字段的具体取值是否适配对应的模型，benchmark本身不会检测，在推理时acl接口报错不属于benchmark的问题。
### 1. 静态shape场景示例，以resnet18模型为例
#### atc命令转换出带动态aipp配置的静态shape模型
```
atc --framework=5 --model=./resnet18.onnx --output=resnet18_bs4_dym_aipp --input_format=NCHW --input_shape="image:4,3,224,224" --soc_version=<soc_version> --insert_op_conf=dym_aipp_conf.aippconfig
```
- dym_aipp_conf.aippconfig的内容(下同)为：
```
aipp_op{
    related_input_rank : 0
    aipp_mode : dynamic
    max_src_image_size : 4000000
}
```
#### benchmark命令
```
msit benchmark --om-model resnet18_bs4_dym_aipp.om --aipp-config actual_aipp_conf.config
```
### 2. 动态batch场景示例，以resnet18模型为例
#### atc命令转换出带动态aipp配置的动态batch模型
```
atc --framework=5 --model=./resnet18.onnx --output=resnet18_dym_batch_aipp --input_format=NCHW --input_shape="image:-1,3,224,224" --dynamic_batch_size "1,2" --soc_version=<soc_version> --insert_op_conf=dym_aipp_conf.aippconfig
```
#### benchmark命令
```
msit benchmark --om-model resnet18_dym_batch_aipp.om --aipp-config actual_aipp_conf.config --dym-batch 1
```
### 3. 动态宽高场景示例，以resnet18模型为例
#### atc命令转换出带动态aipp配置的动态宽高模型
```
atc --framework=5 --model=./resnet18.onnx --output=resnet18_dym_image_aipp --input_format=NCHW --input_shape="image:4,3,-1,-1" --dynamic_image_size "112,112;224,224" --soc_version=<soc_version> --insert_op_conf=dym_aipp_conf.aippconfig
```
#### benchmark命令
```
msit benchmark --om-model resnet18_dym_image_aipp.om --aipp-config actual_aipp_conf.config --dym-hw 112,112
```

## FAQ
使用出现问题时，可参考[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)