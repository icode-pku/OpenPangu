# Caffe 一键式精度比对

## 介绍  
支持标杆模型为Caffe的一键式精度比对。

## 使用示例一
- caffe环境安装请参考：[msit一体化工具使用指南](https://gitcode.com/Ascend/msit/blob/master/msit/docs/install/README.md)
- 对于 Caffe 模型，目前不支持动态 shape 的模型比对。对于 `yolov2` / `yolov3` / `ssd` 等需要自定义实现层的模型，需要自行编译安装特定版本的 caffe
- 准备 Caffe 模型结构文件与权重文件，示例模型结构文件参照 [Caffe 模型结构文件示例](#caffe-模型结构文件示例)，以下使用该文件定义模型，并保存随机初始化的权重，实际使用中应使用已有的模型结构文件 `.prototxt` 与权重文件 `.caffemodel`
  ```py
  import caffe

  model_path = "caffe_demo.prototxt"
  weight_path = "caffe_demo.caffemodel"

  net = caffe.Net(model_path, caffe.TEST)
  net.save(weight_path)
  ```
- 使用 `atc` 命令将 caffe 转化为 om 模型
  ```sh
  atc --model=caffe_demo.prototxt --weight=caffe_demo.caffemodel --framework=0 --soc_version=<soc_version> --output=caffe_demo
  ```
- 注：执行量化原始模型（GPU/CPU） vs 量化离线模型 （**关闭融合规则**）（NPU） 。由于ATC工具的模型转换操作默认开启了算子融合功能，故为了排除融合后算子无法直接进行精度比对，模型转换先关闭算子融合，配置方法参见：[如何关闭/开启融合规则](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/graph/graphubfusionref/atlasrr_30_0003.html#ZH-CN_TOPIC_0000002263813281__section15766181519012)
  ```sh
  atc --model=resnet50_deploy_model.prototxt --weight=resnet50_deploy_weights.caffemodel --framework=0   \
  --output=caffe_resnet50_off --soc_version=<soc_version>  --fusion_switch_file=fusion_switch.cfg
  ```  
  

说明： 关闭算子融合功能需要通过`--fusion_switch_file`参数指定算子融合规则配置文件（如fusion_switch.cfg），并在配置文件中关闭算子融合。 融合规则配置文件关闭配置如下：
```
{
    "Switch":{
        "GraphFusion":{
            "ALL":"off"
        },
        "UBFusion":{
            "ALL":"off"
         }
    }
}
```

- **精度比对调用**
  ```sh
  mkdir -p test  # dump 数据以及比对结果输出路径
  ASCEND_TOOLKIT=$HOME/Ascend/ascend-toolkit/latest  # 必须为可写的 CANN 包路径
  msit debug compare -gm caffe_demo.prototxt -w caffe_demo.caffemodel -om caffe_demo.om -c $ASCEND_TOOLKIT -o ./test
  # ...
  # [INFO] Caffe input info:
  # [{'name': 'data', 'shape': (1, 3, 32, 32), 'type': 'float32'}]
  # ...
  # [INFO] b'2023-05-31 05:56:20 (2749324) - [INFO] The command was completed and took 0 seconds.'
  # [INFO] Compare Node_output:0 completely.
  # [INFO] Analyser init parameter csv_path=~/workspace/caffe_dump/test/20230531055610/result_20230531055619.csv
  # [INFO] Analyser call parameter strategy=FIRST_INVALID_OVERALL, max_column_len=30
  # [INFO] None operator with accuracy issue reported
  ```
- **输出目录结构** 参考 [01_basic_usage](../01_basic_usage/README.md)，其中 caffe 模型 dump 数据位于 `{output_path}/{timestamp}/dump_data/caffe/`
- **比对结果** 位于 `{output_path}/{timestamp}/result_{timestamp}.csv` 中，比对结果的含义与基础精度比对工具完全相同，其中每个字段的含义可参考 [完整比对结果参数说明](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/modelaccuracy/atlasaccuracy_16_0064.html)


## 使用示例二
- Caffe非量化原始模型 vs 量化离线模型场景时，使用 `atc` 命令将**量化**后caffe 转化为 om 模型：
  ```sh
  atc --model=caffe_demo.prototxt --weight=caffe_demo.caffemodel --framework=0 --soc_version=<soc_version> --output=caffe_demo
  ```
- 再转出json文件：
  ```sh
   atc --om caffe_demo.om --mode 1 --json=caffe_demo.json
  ```
- **精度比对调用**
- 使用**量化前**caffe模型：
  ```sh
  mkdir -p test  # dump 数据以及比对结果输出路径
  ASCEND_TOOLKIT=$HOME/Ascend/ascend-toolkit/latest  # 必须为可写的 CANN 包路径
  msit debug compare -gm ResNet-50-deploy.prototxt -w ResNet-50-model.caffemodel -om caffe_demo.om -c $ASCEND_TOOLKIT -o ./test -q caffe_demo.json
  ```

## Caffe 模型结构文件示例
- `caffe_demo.prototxt`
```java
name: "caffe_Demo"
layer {
  name: "Input_1"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 32
      dim: 32
    }
  }
}

layer {
        bottom: "data"
        top: "conv1"
        name: "conv1"
        type: "Convolution"
        convolution_param {
                num_output: 64
                kernel_size: 7
                pad: 3
                stride: 2
        }
}

layer {
        bottom: "conv1"
        top: "conv1"
        name: "bn_conv1"
        type: "BatchNorm"
        batch_norm_param {
                use_global_stats: true
        }
}

layer {
        bottom: "conv1"
        top: "conv1"
        name: "conv1_relu"
        type: "ReLU"
}

layer {
        bottom: "conv1"
        top: "pool5"
        name: "pool5"
        type: "Pooling"
        pooling_param {
                kernel_size: 16
                stride: 1
                pool: AVE
        }
}

layer {
        bottom: "pool5"
        top: "fc1000"
        name: "fc1000"
        type: "InnerProduct"
        inner_product_param {
                num_output: 1000
        }
}
```
