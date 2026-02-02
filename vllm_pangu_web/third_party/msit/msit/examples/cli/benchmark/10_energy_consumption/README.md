# Energy Consumption

## 介绍

指定Device所对应的NPU ID进行推理测试，并获取推理能耗数据。

## 运行示例
查询目前所有芯片的映射关系信息，命令如下：
```bash
npu-smi info -m
```
输出结果示例如下：
```bash
NPU ID                         Chip ID                        Chip Logic ID                  Chip Name
4                              0                              0                              <soc_version>
4                              1                              -                              Mcu
```
输出说明：

| 字段      | 说明                                                         |
|----------------| ------------------------------------------------------------ |
| NPU ID             | 设备id |
| Chip ID            | 芯片id |
| Chip Logic ID      | 芯片逻辑id |
| Chip Name          | 芯片名称 |

其中device 0所对应的NPU ID为4。
```bash
msit benchmark --om-model ./pth_resnet50_bs1.om --device 0 --energy-consumption 1 --npu-id 4
```

输出结果示例如下：

```bash
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 2.4769999980926514, max = 3.937000036239624, mean = 3.5538000106811523, median = 3.7230000495910645, percentile(99%) = 3.936680030822754
[INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(3.5538000106811523): 281.38893494131406
[INFO] ------------------------------------------------------
[INFO] NPU ID:4 energy consumption(J):59.88656545251415
[INFO] unload model success, model Id is 1
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

其中结果最后展示指定Device所对应的NPU ID进行模型推理所消耗的能耗数据energy consumption(J)，单位为焦耳(J)。

## FAQ
使用出现问题时，可参考[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)
