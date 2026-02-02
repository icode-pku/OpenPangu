# 推理组件介绍

## 简介

inference_engine是一个包括推理全流程的组件库，推理端到端流程包含4个引擎，具体引擎如下所示：

- [x] dataset：数据集引擎
- [x] pre process：数据预处理引擎
- [x] inference：离线推理引擎
- [x] post process：后处理引擎
- [x] evaluate：精度评测引擎

组件通过python多进程实现并行化离线推理，组件间通过队列实现数据通信，数据格式为numpy。

## 软件架构

组件采用注册机制，通过factory类提供的add接口实现引擎的注册，具体参考[data_process_factory.py](./data_process_factory.py)文件。

![软件架构](../../docs/img/inference.png)

## 数据传输约束

数据队列建议存放数据格式：[[batch_label], [[batch_data_0], [batch_data_1], [batch_data_n]]]

- [x] batch_label：表示多batch时，对应标签
- [x] batch_data_n：表示第n个输出，batch_data_n包含batch组数据
- [x] 数据格式为numpy

## 数据集引擎

### 数据集介绍

对数据集进行数据处理，处理后的数据满足推理引擎要求。输出数据格式需满足数据格式要求。

示例代码参考[imagenet.py](datasets/vision/imagenet.py)

### 数据集API

- [x] 数据集注册接口

```python
def DatasetFactory.register(name, dataset)
# name: 数据集名称，名称唯一，不能重复
# dataset: 数据集实现类，继承DatasetBase类
```

- [x] 数据集获取接口

```python
def DatasetFactory.get_dataset(name)
# name：数据集名称
```

## 预处理引擎

### 预处理介绍

对数据集进行数据处理，处理后的数据满足推理引擎要求。输出数据格式需满足数据格式要求。

示例代码参考[classification.py](pre_process/vision/classification.py)

### 预处理API

- [x] 预处理注册接口

```python
def PreProcessFactory.register(name, pre_process)
# name: 预处理名称，名称唯一，不能重复
# pre_process：预处理实现类，继承PreProcessBase类
```

- [x] 预处理获取接口

```python
def PreProcessFactory.get_pre_process(name)
# name: 预处理名称

```

## 推理引擎

### 推理介绍

离线推理，输入预处理的数据，执行模型输出得到输出结果。

推理引擎包括如下两种方式：

- [x] onnxruntime离线推理
- [x] 昇腾pyacl离线推理

示例代码参考[onnx_inference.py](./inference/onnx_inference.py)

### 推理API

- [x] 推理注册接口

```python
def InferenceFactory.register(name, inference)
# name: 推理名称，名称唯一，不能重复（比如acl、onnx推理）
# inference：推理实现类，继承InferenceBase类
```

- [x] 推理获取接口

```python
def InferenceFactory.get_inference(name)
# name: 推理名称

```

## 后处理引擎

### 后处理介绍

大部分场景下后处理无需操作，数据直接透传接口。有一些场景下需要先对数据处理后再送精度评测引擎。
比如YOLO系列，大部分场景需要先对3层feature map处理（nms等）后再送精度评测引擎

示例代码参考[classification.py](./post_process/vision/classification.py)

### 后处理API

- [x] 后处理注册接口

```python
def PostProcessFactory.register(name, post_process)
# name: 后处理名称，名称唯一，不能重复
# post_process：后处理实现类，继承PostProcessBase类
```

- [x] 后处理获取接口

```python
def PostProcessFactory.get_post_process(name)
# name：后处理名称
```

## 精度评测引擎

### 精度评测介绍

从后处理得到的数据（包括推理数据和文件名称），利用官方数据提供的功能实现数据集的评测。
示例代码参考[classification.py](./evaluate/vision/classification.py)

### 精度评测API

- [x] 精度评测注册接口

```python
def EvaluateFactory.register(name, evaluate)
# name: 精度评测名称，名称唯一，不能重复
# evaluate：精度评测实现类，继承EvaluateBase类
```

- [x] 精度评测获取接口

```python
def EvaluateFactory.get_evaluate(name)
# name: 精度评测名称
```
