# Caffe Dump

## 介绍
支持指定caffe模型进行tensor数据dump。

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

- **精度比对调用**
  ```sh
  mkdir -p test  # dump 数据以及比对结果输出路径
  ASCEND_TOOLKIT=$HOME/Ascend/ascend-toolkit/latest  # 必须为可写的 CANN 包路径
  msit debug dump -m caffe_demo.prototxt -w caffe_demo.caffemodel -dp cpu -c $ASCEND_TOOLKIT -o ./test
  ```
- **输出目录结构** 参考 [01_basic_usage](../01_basic_usage/README.md)，其中 caffe 模型 dump 数据位于 `{output_path}/{timestamp}/dump_data/caffe/`

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
