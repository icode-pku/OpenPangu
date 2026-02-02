# Save Output Data


## 介绍

默认情况下，dump数据保存在当前目录下的timestamp(例如./20230601115623)目录中。可使用-o或--output指定输出目录。

## 运行示例

```sh
  msit debug dump -m /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -dp cpu
  -i /home/HwHiAiUser/result/test/input_0.bin -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
```
该场景下，输出会保存在/home/HwHiAiUser/result/test/{timestamp}中
