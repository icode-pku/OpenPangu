# Specify Input Data


## 介绍
如果用户想要指定模型输入文件，支持通过参数`--input`指定
- **注意**：如果为文件夹输入，请确保使用的所有输入文件均为.bin后缀文件。

## 运行示例

**指定模型输入** 命令示例，**其中路径需使用绝对路径**
```sh
msit debug dump -m /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -dp cpu
-i /home/HwHiAiUser/result/test/input_0.bin -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
```
  - `-i, –-input` 模型的输入数据路径，默认根据模型的 input 随机生成，多个输入以逗号分隔。例如：`/home/input_0.bin,/home/input_1.bin`，推理时会根据输入shape和模型定义shape进行计算得到batch大小，但需保证输入文件的shape和模型定义的输入shape仅在batch维度不一致，其他维度需保持一致。
```sh
msit debug dump -m /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -dp cpu
-i /home/HwHiAiUser/result/test/input_0.npy -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
```
- `-i, –-input` 如果存在npy文件，支持自动将npy文件转化为bin文件，而不需要手动转化为bin文件，例如：`/home/input_0.npy,/home/input_1.npy`
