# Basic Usage


## 介绍
benchmark推理工具可以通过msit命令行方式启动模型测试。


## 运行示例
### 1. 纯推理场景。
**默认情况下，构造全为0的数据送入模型推理，输出信息仅打屏显示。**
- 输入命令

    ```bash
    msit benchmark --om-model *.om
    ```

  其中，*为OM离线模型文件名。
- 以单个batch的静态模型`resnet50_bs1.om`为例进行推理，执行过程如下所示：

    ```
    [INFO] acl init success
    [INFO] open device 0 success
    [INFO] load model pth_resnet50_bs1.om success
    [INFO] create model description success
    [INFO] try get model batchsize:1
    [INFO] warm up 1 done
    Inference array Processing: 100%|████████████████████████████████| 1/1 [00:00<00:00, 10.73it/s]
    [INFO] -----------------Performance Summary------------------
    [INFO] NPU_compute_time (ms): min = 2.4560000896453857, max = 2.4560000896453857, mean = 2.4560000896453857, median = 2.4560000896453857, percentile(99%) = 2.4560000896453857
    [INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(2.4560000896453857): 407.16610891670894
    [INFO] ------------------------------------------------------
    [INFO] unload model success, model Id is 1
    [INFO] end to destroy context
    [INFO] end to reset device is 0
    [INFO] end to finalize acl
    ```
  - 打屏信息解释：
  ```
  NPU_compute_time (ms): # 推理时间，不包括H2D（host to device）和D2H（device to host）的时间
      min = 2.4560000896453857 # 推理的最短时间
      max = 2.4560000896453857 # 推理的最长时间
      mean = 2.4560000896453857 # 推理的平均时间
      median = 2.4560000896453857 # 推理时间中位数
  ```
  ```
  throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(2.4560000896453857): 407.16610891670894 # 推理的吞吐率，计算公式为1000*batchsize.mean(1)/NPU_compute_time.mean(2.4560000896453857)
  ```

### 2. 调试模式。
  **开启debug调试模式。**

  ```bash
  msit benchmark --om-model /home/model/resnet50_v1.om --output ./ --debug 1
  ```

  调试模式开启后会增加更多的打印信息，包括：
   - 模型的输入输出参数信息

     ```bash
      [INFO] try get model batchsize:1
      [DEBUG] Input nums: 1
      [DEBUG] Model id: 1
      [DEBUG] aipp_input_exist: 0
      [DEBUG] session info:<Model>
      device:	0
      input:
        #0    actual_input_1  (1, 3, 224, 224)  float32  602112  602112
      output:
        #0    PartitionedCall_/fc/Gemm_add_4:0:output1  (1, 1000)  float32  4000  4000
     ```

   - 详细的推理耗时信息

     ```bash
     [DEBUG] model aclExec cost : 2.336000
     ```
   - 模型输入输出等具体操作信息

## FAQ
使用出现问题时，可参考[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)