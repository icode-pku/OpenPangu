# Dynamic grading


## 介绍

动态分档场景。主要包含动态Batch、动态HW（宽高）、动态Dims三种场景，需要分别传入--dym-batch、--dym-hw、--dym-dims指定实际档位信息。

## 运行示例

1. 动态Batch。

    以档位1 2 4 8档为例，设置档位为2，本程序将获取实际模型输入组Batch，每2个输入为一组，进行组Batch。

    ```bash
    msit benchmark --om-model ./resnet50_v1_dynamicbatchsize_fp32.om --input=./data/ --dym-batch 2
    ```

2. 动态HW宽高。

    以档位224,224;448,448档为例，设置档位为224,224，本程序将获取实际模型输入组Batch。

    ```bash
    msit benchmark --om-model ./resnet50_v1_dynamichw_fp32.om --input=./data/ --dym-hw 224,224
    ```

3. 动态Dims。

   以设置档位1,3,224,224为例，本程序将获取实际模型输入组Batch。

   ```bash
   msit benchmark --om-model resnet50_v1_dynamicshape_fp32.om --input=./data/ --dym-dims actual_input_1:1,3,224,224
   ```

4. 自动设置Dims模式（动态Dims模型）。

    动态Dims模型输入数据的Shape可能是不固定的，比如一个输入文件Shape为1,3,224,224，另一个输入文件Shape为 1,3,300,300。若两个文件同时推理，则需要设置两次动态Shape参数，当前不支持该操作。针对该场景，增加--auto-set-dymdims-mode模式，可以根据输入文件的Shape信息，自动设置模型的Shape参数。

    ```bash
    msit benchmark --om-model resnet50_v1_dynamicshape_fp32.om --input=./data/ --auto-set-dymdims-mode 1
    ```
- 说明：示例中的./data/文件夹存放用户的输入数据，为.npy格式，若不指定输入数据，会自动生成随机的输入数据。
## FAQ
使用出现问题时，可参考[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)