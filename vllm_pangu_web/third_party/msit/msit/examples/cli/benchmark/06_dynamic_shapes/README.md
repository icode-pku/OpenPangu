# Dynamic shapes


## 介绍

动态shape场景。主要包含动态Shape、自动设置Shape模式（动态Shape模型）、动态Shape模型range测试模式三种场景，需要分别传入--dym-shape、--auto-set-dymshape-mode、--dym-shape-range指定动态shape信息。

## 运行示例

1. 动态Shape场景。

    以ATC设置[1\~8,3,200\~300,200\~300]，设置档位1,3,224,224为例，本程序将获取实际模型输入组Batch。

    动态Shape的输出大小通常为0，建议通过output-size参数设置对应输出的内存大小。

    ```bash
    msit benchmark --om-model resnet50_v1_dynamicshape_fp32.om --dym-shape actual_input_1:1,3,224,224 --output-size 10000
    ```

2. 自动设置Shape模式（动态Shape模型）。

    动态Shape模型输入数据的Shape可能是不固定的，比如一个输入文件Shape为1,3,224,224 另一个输入文件Shape为 1,3,300,300。若两个文件同时推理，则需要设置两次动态Shape参数，当前不支持该操作。针对该场景，增加--auto-set-dymshape-mode模式，可以根据输入文件的Shape信息，自动设置模型的Shape参数。

    ```bash
    msit benchmark --om-model ./pth_resnet50_dymshape.om  --output-size 100000 --auto-set-dymshape-mode 1  --input ./dymdata
    ```

    **注意该场景下的输入文件必须为npy格式，如果是bin文件将获取不到真实的Shape信息。**

3. 单输入动态Shape模型range测试模式。

    输入动态Shape的range范围。对于该范围内的Shape分别进行推理，得出各自的性能指标。

    以对1,3,224,224 1,3,224,225 1,3,224,226进行分别推理为例，命令如下：

    ```bash
    msit benchmark --om-model ./pth_resnet50_dymshape.om  --output-size 100000 --dym-shape-range actual_input_1:1,3,224,224~226
    ```

4. 多输入动态Shape模型range测试模式（命令行）。

    输入动态Shape的range范围。对于该范围内的Shape分别进行推理，得出各自的性能指标。

    以一个双输入模型为例，第一个输入为1,3,224,224 1,3,224,225 1,3,224,226，第二个输入为1,3,224,224 2,3,224,224 3,3,224,224，进行分别推理为例，命令如下：

    ```bash
    msit benchmark --om-model ./pth_resnet50_dymshape_dual_input.om  --output-size 100000 --dym-shape-range "actual_input_1:1,3,224,224~226;actual_input_2:1~3,3,224,224"
    ```

5. 多输入动态Shape模型range测试模式（*.info）。

    输入动态Shape的range范围。对于该范围内的Shape分别进行推理，得出各自的性能指标。

    以一个双输入模型为例，对两组输入分别进行推理。
    <br/>第一组输入为：
    <br/>第一个输入为1,3,224,224 1,3,224,225 1,3,224,226，第二个输入为1,3,224,224 2,3,224,224 3,3,224,224。
    <br/>第二组输入为：
    <br/>第一个输入为1,3,224,224 2,3,224,224 3,3,224,224，第二个输入为1,3,224,224 1,3,224,225 1,3,224,226。
    <br/>首先创建一个info文件，命名为dual_input.info，内容为：
   <br/> actual_input_1:1,3,224,224\~226;actual_input_2:1\~3,3,224,224
   <br/> actual_input_1:1\~3,3,224,224;actual_input_2:1,3,224,224\~226
    
    命令如下：

    ```bash
    msit benchmark --om-model ./pth_resnet50_dymshape_dual_input.om  --output-size 100000 --dym-shape-range dual_input.info
    ```
    >注：actual_input为模型实际的输入名称
## FAQ
使用出现问题时，可参考[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)