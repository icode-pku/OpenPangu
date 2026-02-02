# Aipp Model Compare


## 介绍

支持对原始onnx模型和开启aipp选项后转换得到的离线OM模型进行精度比对。

## 运行示例

### 准备工作
先使用[atc工具](https://www.hiascend.com/document/detail/zh/canncommercial/800/devaids/devtools/atc/atlasatc_16_0005.html)重新转换一个算子不融合的om模型：
```sh
atc --framework 5 --model=./resnet18.onnx --output=resnet18_bs8 --input_format=NCHW \
--input_shape="image:8,3,224,224" --log=debug --soc_version=<soc_version> \
--insert_op_conf=aipp.config --fusion_switch_file=fusionswitch.cfg
```
其中fusionswitch.cfg(算子不融合)内容如下：
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
aipp.config内容样例如下：
```
aipp_op{
    aipp_mode:static
    input_format : RGB888_U8

    src_image_size_w : 256
    src_image_size_h : 256

    crop: true
    load_start_pos_h : 16
    load_start_pos_w : 16
    crop_size_w : 224
    crop_size_h: 224

    min_chn_0 : 123.675
    min_chn_1 : 116.28
    min_chn_2 : 103.53
    var_reci_chn_0: 0.0171247538316637
    var_reci_chn_1: 0.0175070028011204
    var_reci_chn_2: 0.0174291938997821
}
```

### 命令行操作
```sh
msit debug compare -gm ./resnet18.onnx -om ./resnet18_bs8.om -is "image:8,3,224,224"
```
-gm为标杆onnx模型(**必选**)；-om参数请输入上述生成的算子不融合的om模型(**必选**)；-is为onnx模型输入的shape信息(**必选**)；如果需要指定输入(可选)，请使用-i参数指定om模型的输入(npy或者bin文件)。