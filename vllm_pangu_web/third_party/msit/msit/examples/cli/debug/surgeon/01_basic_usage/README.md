# Basic Usage

## 介绍

surgeon 工具可通过 msit 命令行形式启动。

## 运行示例

```bash
msit debug surgeon COMMAND [OPTIONS] [REQUIRED]
```

其中*COMMAND*为 surgeon 执行模式参数，取值为 list、evaluate、optimize、extract和concatenate；[OPTIONS]和[REQUIRED]为可选项和必选项参数，每个子任务下面的可选项和必选项不同。

## 使用流程

surgeon 工具建议按照 list、evaluate 和 optimize 的顺序执行。如需切分子图，可使用 extract 命令导出子图。

操作流程如下：

1. 执行**list**命令列举当前支持自动调优的所有知识库。
2. 执行**evaluate**命令搜索可以被指定知识库优化的 ONNX 模型。
3. 执行**optimize**命令使用指定的知识库来优化指定的 ONNX 模型。
4. 执行**extract**命令对模型进行子图切分。
5. 执行**concatenate**命令拼接模型。
