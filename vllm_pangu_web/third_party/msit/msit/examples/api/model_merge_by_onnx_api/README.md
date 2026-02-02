# ONNX 合并工具介绍
ONNX合并工具可以将两个ONNX文件合并为一个，可通过多次使用将若干ONNX最终合并为一个ONNX文件。

该工具基于原始的 onnx.compose 功能，消除了自定义算子check、ir版本的限制，并进行了参数的封装。

工具的命令行使用示例为：
```bash
python combine.py --previous_model_path sub_model1.onnx --following_model_path sub_model2.onnx --merge_model_path merged_model.onnx --previous_model_outputs output1 --following_model_inputs input1
```

包含五个参数：
1. previous_model_path  在拓扑图中靠前的子图A的ONNX文件路径
2. following_model_path   在拓扑图中靠后的子图B的ONNX文件路径
3. merge_model_path  合并后的ONNX文件路径
4. previous_model_outputs  子图A的outputs名称
5. following_model_inputs  子图B的inputs名称

合并工具将 **A的outputs** 和 **B的inputs** 一一对应，进行拼接。outputs和inputs名称可通过可视化工具获取。

以使用示例中的参数为例：sub_model1.onnx 的 output1 将和 sub_model2.onnx 的 input1 进行连接，从而将 sub_model1.onnx
 和 sub_model2.onnx 合并为一个文件。若子网存在多个 input、output，可在previous_model_outputs、following_model_inputs参数中输入多个名称，例如
```bash
--previous_model_outputs output1 output2 --following_model_inputs input1 input2
```