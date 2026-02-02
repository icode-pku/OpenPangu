# msIT

## 介绍

MindStudio Inference Tools，昇腾推理工具链。 【Powered by MindStudio】

**请根据自己的需要进入对应文件夹获取工具，或者点击下面的说明链接选择需要的工具进行使用。**

### 模型推理迁移全流程
![模型推理迁移全流程](./msit-flow.png)

### 大模型推理迁移全流程
![大模型推理迁移全流程](./msit-llm-flow.png)

## 使用说明

1.  [msit](https://gitcode.com/Ascend/msit/tree/master/msit)

    **一体化推理开发工具**：作为昇腾统一推理工具，提供客户一体化开发工具，支持一站式调试调优，当前包括benchmark、debug、analyze等组件。

2.  [msmodelslim](https://gitcode.com/Ascend/msit/tree/master/msmodelslim)

    **昇腾模型压缩加速工具**：一个以加速为目标、压缩为技术、昇腾为根本的亲和压缩工具。支持训练加速和推理加速，包括模型低秩分解、稀疏训练、训练后量化、量化感知训练等功能，昇腾AI模型开发用户可以灵活调用Python API接口，对模型进行性能调优，并支持导出不同格式模型，在昇腾AI处理器上运行。

3.  [msserviceprofiler](https://gitcode.com/Ascend/msit/tree/master/msserviceprofiler)

    **推理服务化性能调优工具**：msserviceprofiler 是一款基于昇腾平台，支持MindIE Service框架和vLLM框架的服务化调优工具。 其性能采集与数据解析能力已嵌入昇腾CANN工具包，支持MindStudio Insight、Chrome Tracing、Grafana多个平台数据可视化。目前包括analyse、split、compare、advisor、optimizer和train等组件

4.  [msprechecker](https://gitcode.com/Ascend/msit/tree/master/msprechecker)
    
    **预检工具**：msprechecker 提供推理场景的预检能力，支持环境预检，连通性预检，推理过程中的落盘和比对功能。帮助用户在推理业务部署前，提前发现异常问题。推理时，提高推理性能，快速复现基线。
#### 许可证
[Apache License 2.0](LICENSE)
