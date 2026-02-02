
# trtexec场景

benchmark支持onnx模型推理（集成trtexec）,trtexec为NVIDIA TensorRT自带工具。用户使用benchmark拉起trtexec工具进行推理性能测试，测试过程中实时输出trtexec日志，打印在控制台，推理性能测试完成后，将性能数据输出在控制台。
## 前置条件
推理性能测试环境需要配置有GPU，安装CANN、CUDA及TensorRT，并且trtexec可以通过命令行调用到，安装方式可参考[TensorRT](https://github.com/NVIDIA/TensorRT)。

示例命令如下：

```bash
msit benchmark -om pth_resnet50.onnx --backend trtexec --perf 1

```

输出结果推理测试结果，示例如下：

```bash
[INFO] [05/27/2023-12:05:31] [I] === Performance summary ===
[INFO] [05/27/2023-12:05:31] [I] Throughput: 120.699 qps
[INFO] [05/27/2023-12:05:31] [I] Latency: min = 9.11414 ms, max = 11.7442 ms, mean = 9.81005 ms, median = 9.76404 ms, percentile(90%) = 10.1075 ms, percentile(95%) = 10.1624 ms, percentile(99%) = 11.4742 ms
[INFO] [05/27/2023-12:05:31] [I] Enqueue Time: min = 0.516296 ms, max = 0.598633 ms, mean = 0.531443 ms, median = 0.5271 ms, percentile(90%) = 0.546875 ms, percentile(95%) = 0.564575 ms, percentile(99%) = 0.580566 ms
[INFO] [05/27/2023-12:05:31] [I] H2D Latency: min = 1.55066 ms, max = 1.57336 ms, mean = 1.55492 ms, median = 1.55444 ms, percentile(90%) = 1.55664 ms, percentile(95%) = 1.55835 ms, percentile(99%) = 1.56458 ms
[INFO] [05/27/2023-12:05:31] [I] GPU Compute Time: min = 7.54407 ms, max = 10.1723 ms, mean = 8.23978 ms, median = 8.19409 ms, percentile(90%) = 8.5354 ms, percentile(95%) = 8.59131 ms, percentile(99%) = 9.90002 ms
[INFO] [05/27/2023-12:05:31] [I] D2H Latency: min = 0.0130615 ms, max = 0.0170898 ms, mean = 0.015342 ms, median = 0.0153809 ms, percentile(90%) = 0.0162354 ms, percentile(95%) = 0.0163574 ms, percentile(99%) = 0.0168457 ms
[INFO] [05/27/2023-12:05:31] [I] Total Host Walltime: 3.02405 s
[INFO] [05/27/2023-12:05:31] [I] Total GPU Compute Time: 3.00752 s
```

**字段说明**

| 字段                  | 说明                                                         |
| --------------------- | ------------------------------------------------------------ |
| Throughput            | 吞吐率。                    |
| Latency               | H2D 延迟、GPU 计算时间和 D2H 延迟的总和。这是推断单个执行的延迟。                   |
| min                   | 推理执行时间最小值。                                         |
| max                   | 推理执行时间最大值。                                         |
| mean                  | 推理执行时间平均值。                                         |
| median                | 推理执行时间取中位数。                                       |
| percentile(99%)       | 推理执行时间中的百分位数。                                   |
| H2D Latency           | 单个执行的输入张量的主机到设备数据传输的延迟。                                   |
| GPU Compute Time      | 为执行 CUDA 内核的 GPU 延迟。                                |
| D2H Latency           | 单个执行的输出张量的设备到主机数据传输的延迟。                    |
| Total Host Walltime   | 从第一个执行（预热后）入队到最后一个执行完成的主机时间。 |
| Total GPU Compute Time| 所有执行的 GPU 计算时间的总和。 |

## FAQ
使用出现问题时，可参考[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)