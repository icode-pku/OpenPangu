# Save Output Data

## 介绍

默认情况下，benchmark推理工具执行后不保存输出结果数据文件，配置相关参数后，可生成的结果数据如下：

| 文件/目录                                | 说明                                                                                                                                                                                                                                                                                                         |
| ---------------------------------------- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| {文件名}.bin、{文件名}.npy或{文件名}.txt | 模型推理输出结果文件。<br/>文件命名格式：名称_输出序号.后缀。不指定input时（纯推理），名称固定为“pure_infer_data”；指定input时，名称以第一个输入的第一个名称命名；输出的序号从0开始按输出先后顺序排列；文件名后缀由--outfmt参数控制。<br/>默认情况下，会在--output参数指定的目录下创建“日期+时间”的目录，并将结果文件保存在该目录下；当指定了--output-dirname时，结果文件将直接保存在--output-dirname参数指定的目录下。<br/>指定--output-dirname参数时，多次执行工具推理会导致结果文件因同名而覆盖。 |
| xx_summary.json                          | 工具输出模型性能结果数据。默认情况下，“xx”以“日期+时间”命名；当指定了--output-dirname时，“xx”以--output-dirname指定的目录名称命名。<br/>指定--output-dirname参数时，多次执行工具推理会导致结果文件因同名而覆盖。                                                                                                                                                                   |
| dump                                     | dump数据文件目录。使用--dump开启dump时，在--output参数指定的目录下创建dump目录，保存dump数据文件。                                                                                                                                                                                                                                           |
| profiler                                 | Profiler采集性能数据文件目录。使用--profiler开启性能数据采集时，在--output参数指定的目录下创建profiler目录，保存性能数据文件。                                                                                                                                                                                                                           |

## 运行示例
1. 仅设置--output参数。示例命令及结果如下：

  ```bash
  msit benchmark --om-model ./pth_resnet50_bs1.om --output ./result
  ```

  ```bash
  result
  |-- 2022_12_17-07_37_18
  │   |-- pure_infer_data_0.bin
  |-- 2022_12_17-07_37_18_summary.json
  ```

2. 设置--input和--output参数。示例命令及结果如下：

  ```bash
  # 输入的input文件夹内容如下
  ls ./data
  196608-0.bin  196608-1.bin  196608-2.bin  196608-3.bin  196608-4.bin  196608-5.bin  196608-6.bin  196608-7.bin  196608-8.bin  196608-9.bin
  ```
  - 说明：
    .bin文件存储用户输入的tensor数据，可通过以下方式生成，例子中的size和astype可以通过debug调试模式工具获取。--input参数是为了用户指定输入数据而设计。
    ```python
    import numpy as np
    np.random.uniform(size=[32,32]).astype('float32').tofile('foo.bin')
    ```
      
  ```bash
  msit benchmark --om-model ./pth_resnet50_bs1.om --input ./data --output ./result
  ```

  ```bash
  result/
  |-- 2023_01_03-06_35_53
  |   |-- 196608-0_0.bin
  |   |-- 196608-1_0.bin
  |   |-- 196608-2_0.bin
  |   |-- 196608-3_0.bin
  |   |-- 196608-4_0.bin
  |   |-- 196608-5_0.bin
  |   |-- 196608-6_0.bin
  |   |-- 196608-7_0.bin
  |   |-- 196608-8_0.bin
  |   |-- 196608-9_0.bin
  |-- 2023_01_03-06_35_53_summary.json
  ```

3. 设置--output-dirname参数。示例命令及结果如下：

  ```bash
  msit benchmark --om-model ./pth_resnet50_bs1.om --output ./result --output-dirname subdir
  ```

  ```bash
  result
  |-- subdir
  │   |-- pure_infer_data_0.bin
  |-- subdir_summary.json
  ```

4. 设置--dump参数。示例命令及结果如下：

  ```bash
  msit benchmark --om-model ./pth_resnet50_bs1.om --output ./result --dump 1
  ```

  ```bash
  result
  |-- 2022_12_17-07_37_18
  │   |-- pure_infer_data_0.bin
  |-- dump
  |-- 2022_12_17-07_37_18_summary.json
  ```

5. 设置--profiler参数。示例命令及结果如下：

  ```bash
  msit benchmark --om-model ./pth_resnet50_bs1.om --output ./result --profiler 1
  ```

  ```bash
  result
  |-- 2022_12_17-07_56_10
  │   |-- pure_infer_data_0.bin
  |-- profiler
  │   |-- PROF_000001_20221217075609326_GLKQJOGROQGOLIIB
  |-- 2022_12_17-07_56_10_summary.json
  ```

6. 输出结果解释。

benchmark推理工具执行后，打屏输出结果示例如下：

- --display-all-summary参数设置为False时，打印如下：

  ```bash
  [INFO] -----------------Performance Summary------------------
  [INFO] NPU_compute_time (ms): min = 0.6610000133514404, max = 0.6610000133514404, mean = 0.6610000133514404, median = 0.6610000133514404, percentile(99%) = 0.6610000133514404
  [INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(0.6610000133514404): 1512.8592735267011
  [INFO] ------------------------------------------------------
  ```

- --display-all-summary参数设置为True时，打印如下：

  ```bash
  [INFO] -----------------Performance Summary------------------
  [INFO] H2D_latency (ms): min = 0.05700000002980232, max = 0.05700000002980232, mean = 0.05700000002980232, median = 0.05700000002980232, percentile(99%) = 0.05700000002980232
  [INFO] NPU_compute_time (ms): min = 0.6650000214576721, max = 0.6650000214576721, mean = 0.6650000214576721, median = 0.6650000214576721, percentile(99%) = 0.6650000214576721
  [INFO] D2H_latency (ms): min = 0.014999999664723873, max = 0.014999999664723873, mean = 0.014999999664723873, median = 0.014999999664723873, percentile(99%) = 0.014999999664723873
  [INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(0.6650000214576721): 1503.759349974173
  ```

通过输出结果可以查看模型执行耗时、吞吐率。耗时越小、吞吐率越高，则表示该模型性能越高。

**字段说明**

| 字段                  | 说明                                                    |
| --------------------- |-------------------------------------------------------|
| H2D_latency (ms)      | Host to Device的内存拷贝耗时。单位为ms。                          |
| min                   | 推理执行时间最小值。                                            |
| max                   | 推理执行时间最大值。                                            |
| mean                  | 推理执行时间平均值。                                            |
| median                | 推理执行时间取中位数。                                           |
| percentile(99%)       | 推理执行时间中的百分位数。                                         |
| NPU_compute_time (ms) | NPU推理计算的时间。单位为ms。                                     |
| D2H_latency (ms)      | Device to Host的内存拷贝耗时。单位为ms。                          |
| throughput            | 吞吐率。吞吐率计算公式：1000 *batchsize/npu_compute_time.mean     |
| batchsize             | 批大小。本工具不一定能准确识别当前样本的batchsize，建议通过--batch-size参数进行设置。 |

## FAQ
使用出现问题时，可参考[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)