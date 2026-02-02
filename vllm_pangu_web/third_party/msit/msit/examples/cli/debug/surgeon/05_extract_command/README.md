# Extract Command


## 介绍
对模型进行子图切分。

```bash
msit debug surgeon extract [OPTIONS] [REQUIRED]
```

extract 可简写为ext

参数说明：

| 参数                    | 说明                                                                                                                                                                                                                                                                                                                                 | 是否必选 |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| OPTIONS               | 额外参数。可取值：<br/>    -ck/--is-check-subgraph：是否校验子图。启用这个选项时，会校验切分后的子图。<br/>    -sis/--subgraph-input-shape：额外参数。可指定截取子图之后的输入shape。多节点的输入shape指定按照以下格式，"input1:n1,c1,h1,w1;input2:n2,c2,h2,w2"。<br/>    -sit/--subgraph_input_dtype：额外参数。可指定截取子图之后的输入dtype。多节点的输入dtype指定按照以下格式，"input1:dtype1;input2:dtype2"。<br/>  -h,  --help：工具使用帮助信息。 | 否    |
| REQUIRED              | -in/--input：输入ONNX待优化模型，必须为.onnx文件。 <br/>    -of/--output-file：切分后的子图ONNX模型名称，用户自定义，必须为.onnx文件。<br/>    -snn/--start-node-names：起始节点名称。可指定多个输入节点，节点之间使用","分隔。<br/>     -enn/--end-node-names：结束节点名称。可指定多个输出节点，节点之间使用","分隔。                                                                                                         | 是    |


使用特别说明：为保证子图切分功能正常使用且不影响推理性能，请勿指定存在**父子关系**的输入或输出节点作为切分参数。

## 运行示例

```bash
msit debug surgeon extract --input=origin_model.onnx --output-file=sub_model.onnx --start-node-names="s_node1,s_node2" --end-node-names="e_node1,e_node2" --subgraph_input_shape="input1:1,3,224,224" --subgraph_input_dtype="input1:float16"
```

输出示例如下：

```bash
2023-04-27 14:32:33,378 - 984068 - msit_debug_logger - INFO - Extract the model completed, model was saved in sub_model.onnx
```