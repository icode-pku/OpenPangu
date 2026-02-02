# Evaluate Command

## 介绍
搜索可以被指定知识库优化的ONNX模型。

```bash
msit debug surgeon evaluate [OPTIONS] PATH
```

evaluate可简写为eva。

参数说明：

| 参数        | 说明                                                                                                                                                                                                                                                           | 是否必选 |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| -------- |
| OPTIONS   | 额外参数。可取值：<br/>    -know/--knowledges：知识库列表。可指定知识库名称或序号，以英文逗号“,”分隔。默认启用除修复性质以外的所有知识库。<br/>    -r/--recursive：在PATH为文件夹时是否递归搜索。默认关闭。<br/>    -v/--verbose：打印更多信息，目前只有搜索进度。默认关闭。<br/>    -p/--processes：使用multiprocess并行搜索，指定进程数量。默认1。<br/>   -h, --help：工具使用帮助信息。 | 否       |
| REQUIRED  | --path：evaluate的搜索目标，可以是.onnx文件或者包含.onnx文件的文件夹。                                                                                                                                                                                                              | 是       |


## 运行示例

```bash
msit debug surgeon evaluate --path=aasist_bs1_ori.onnx
```

输出示例如下：

```
2023-04-27 14:37:10,364 - 984068 - msit_debug_logger - INFO - aasist_bs1_ori.onnx    KnowledgeConv1d2Conv2d,KnowledgeMergeConsecutiveSlice,KnowledgeTransposeLargeInputConv,KnowledgeTypeCast,KnowledgeMergeCasts
```