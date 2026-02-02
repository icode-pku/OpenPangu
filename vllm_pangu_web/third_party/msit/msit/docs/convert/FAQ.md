- [FAQ](#faq)
  - [1. 使用msit convert atc 转换时报错（ONNX格式）](#1-使用msit-convert-atc-转换时报错onnx格式)

# FAQ

----------------------------------------

## 1. 使用msit convert atc 转换时报错（ONNX格式）
**报错提示** E16005：The model has [2] [--domain version] fields, but only one is allowed

**报错原因** 由第三方库开发等原因引入的包含自定义算子的模型，可能使ONNX模型中出现多个domain version，从而导致ATC模型转换报错

**解决方案** 使用surgeon组件对模型中所有节点进行遍历，将与模型domain version不同的节点的domain version置空。以下脚本供参考：

```python
from auto_optimizer import OnnxGraph

g = OnnxGraph.parse("model.onnx")

for node in g.nodes:
    if node.domain:
        node.domain = ""

g.save("modified_model.onnx")
```