# transform类模型大kernel优化

## 介绍

ATC在做模型转换时，对于transform类模型的decoder结构，支持一种标准pattern的融合pass，将整个attention结构融合成两个算子AttentionLnQKV和AttentionScore，从而提升模型的推理性能。因此，在做transform类模型的推理迁移时，通常会将模型的attention结构通过一定的规则转换成标准pattern，以提升模型的推理性能。对于这种场景，surgeon组件提供了大kernel优化的知识库KnowledgeBigKernel，可以将原始的transform模型attention结构自动转换成标准pattern。

## 使用场景约束

1、大kernel优化场景仅在Atlas推理系列产品上有效；

2、执行大kernel优化时，需要提供模型第一个attention结构的起始节点和结束节点；

3、当前已验证模型：gpt2、bert-base和bert-large；

## 运行原理

1、根据模型第一个attention结构的起始节点和结束节点，截取attention子图，通过该子图构建attention的pattern，在全网中搜索此pattern，获取所有的attention子图；

2、对于每一个attention子图，执行以下操作：

​	2.1 解析attention的参数，包括计算qkv的matmul、add等参数，得到参数params；

​	2.2 初始化一个标准子图；

​	2.3 用params去更新标准子图中的参数；

​	2.4 用标准子图替换原始的attention子图；

3、所有的attention子图替换完成之后，需要做一些后处理操作：

​	3.1 由于标准attention和layernorm的input shape只能是2维，所以需要在第一个layernorm之前插入reshape算子，将input转成2维，在最后一个layernorm之后还需插入reshape节点，将output shape重新转成之前的shape。

​	3.2 标准attention子图中有一个Add节点，是将qk相乘的结果与mask相加，暂且称它为qk_mask_add。qk_mask_add节点是在标准pattern里要求在相加时，input2只能第1维做broadcast操作，其他维度如0，2，3如果与input1的shape不一致，需要expand。

​	3.3 删除graph中一些无用节点和参数，保存模型。

## 运行示例

```shell
msit debug surgeon opt --input=bert-base-chinese.onnx --output-file=bert-base-chinese_opt.onnx -bk -as attention_start_name -ae attention_end_name
```

