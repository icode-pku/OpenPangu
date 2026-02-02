# Custom Ops


## 介绍

本案例针对自定义算子开发场景，基于surgeon组件为用户提供在ONNX图中新增自定义算子等功能。

注：通过本案例指导所添加的自定义算子同样适配surgeon组件改图API的全部功能，但不保证可进行推理测试，本案例旨在为新增自定义算子提供一种简易可行的解决方案。

## 运行示例

以下代码示例展示将ONNX图中一个算子替换为自定义算子的流程，用户可按需仅进行新增自定义算子的操作。

```python
from auto_optimizer import OnnxGraph, OnnxNode

g = OnnxGraph.parse("model.onnx")

# 通过算子名称找到找到待替换的节点
ori_node = g.get_node("ori_node_name", node_type=OnnxNode)

# 新增自定义算子
custom_op = g.add_node(
    "custom_op",          # 算子名称
    "CustomOpType"        # 算子类型 （自定义类型）
)

# 将新增自定义算子接入原算子之后
g.insert_node("ori_node_name", custom_op, refer_index=0) # refer_index=0为在参考算子之后插入

# 删除原算子
g.remove("ori_node_name")

# 保存模型
g.update_map()
g.save("model_with_custom_op.onnx")

```