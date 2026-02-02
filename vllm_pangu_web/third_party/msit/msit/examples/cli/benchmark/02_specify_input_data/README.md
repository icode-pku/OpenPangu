# Specify Input Data

## 介绍

默认情况下，构造全为0的数据送入模型推理。可指定文件输入或者文件夹输入。

## 运行示例
1. 文件输入场景。

    使用--input参数指定模型输入文件，多个文件之间通过“,”进行分隔。

    本场景会根据文件输入size和模型实际输入size进行对比，若缺少数据则会自动构造数据补全，称为组Batch。

    ```bash
    msit benchmark --om-model ./resnet50_v1_bs1_fp32.om --input ./1.bin,./2.bin,./3.bin,./4.bin,./5.bin
    ```
   - 说明：
    .bin文件存储用户输入的tensor数据，可通过以下方式生成，例子中的size和astype可以通过debug调试模式工具获取。--input参数是为了用户指定输入数据而设计。
    ```python
    import numpy as np
    np.random.uniform(size=[32,32]).astype('float32').tofile('foo.bin')
    ```
2. 文件夹输入场景。

    使用--input参数指定模型输入文件所在目录，多个目录之间通过“,”进行分隔。

    本场景会根据文件输入size和模型实际输入size进行组Batch。

    ```bash
    msit benchmark --om-model ./resnet50_v1_bs1_fp32.om --input ./
    ```
   - 说明：
     1.如果输入的./文件夹内无.bin文件，会报错，传入--input参数时要确保./内存在.bin数据；
     2.模型输入需要与传入文件夹的个数一致。

    例如：将模型用netron软件打开，可以查看模型的输入，如bert模型有三个输入，分别为：input_ids、 input_mask、 segment_ids，所以传参必须传入3个文件夹，且三个文件夹分别对应模型的三个输入，顺序要对应。

    - 第一个文件夹“./data/SQuAD1.1/input_ids"，对应模型第一个参数"input_ids"的输入
    - 第二个文件夹"./data/SQuAD1.1/input_mask"，对应第二个参数"input_mask"的输入
    - 第三个文件夹"./data/SQuAD1.1/segment_ids"，对应第三个参数"segment_ids"的输入
      
    ```bash
    msit benchmark --om-model ./save/model/BERT_Base_SQuAD_BatchSize_1.om --input ./data/SQuAD1.1/input_ids,./data/SQuAD1.1/input_mask,./data/SQuAD1.1/segment_ids
    ```

## FAQ
使用出现问题时，可参考[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)
