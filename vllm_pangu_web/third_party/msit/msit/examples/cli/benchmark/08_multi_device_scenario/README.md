# Multi Device Scenario

## 介绍

多Device场景下，可以同时指定多个Device进行推理测试。

## 运行示例

```bash
msit benchmark --om-model ./pth_resnet50_bs1.om --input ./data/ --device 1,2
```

输出结果依次展示每个Device的推理测试结果，示例如下：

```bash
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 2.4769999980926514, max = 3.937000036239624, mean = 3.5538000106811523, median = 3.7230000495910645, percentile(99%) = 3.936680030822754
[INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(3.5538000106811523): 281.38893494131406
[INFO] ------------------------------------------------------
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 3.3889999389648438, max = 3.9230000972747803, mean = 3.616000032424927, median = 3.555000066757202, percentile(99%) = 3.9134000968933105
[INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(3.616000032424927): 276.54867008654026
[INFO] ------------------------------------------------------
[INFO] unload model success, model Id is 1
[INFO] unload model success, model Id is 1
[INFO] end to destroy context
[INFO] end to destroy context
[INFO] end to reset device is 2
[INFO] end to reset device is 2
[INFO] end to finalize acl
[INFO] end to finalize acl
[INFO] multidevice run end qsize:4 result:1
i:0 device_1 throughput:281.38893494131406 start_time:1676875630.804429 end_time:1676875630.8303885
i:1 device_2 throughput:276.54867008654026 start_time:1676875630.8043878 end_time:1676875630.8326817
[INFO] summary throughput:557.9376050278543
```

其中结果最后展示每个Device推理测试的throughput（吞吐率）、start_time（测试启动时间）、end_time（测试结束时间）以及summary throughput（吞吐率汇总）。其他详细字段解释请参见本手册的“输出结果”章节。

## FAQ
使用出现问题时，可参考[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)
