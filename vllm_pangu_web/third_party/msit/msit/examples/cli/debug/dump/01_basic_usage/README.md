# Basic Usage


## 介绍
dump功能可以通过msit命令行方式启动, 通过-m参数指定具体的模型文件，当前支持onnx、om、Tensorflow保存的save_model模型。


## 运行示例
- **不指定模型输入** 命令示例，**其中路径需使用绝对路径**
  ```sh
  msit debug dump -m /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -dp cpu
  -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
  ```
  - `-m, –-model` 指定原始离线模型（.onnx）路径
  - `-dp, --device-pattern` 指定dump的设备类型，支持cpu和npu，目前npu模式下只支持saved_model模型
  - `-c，–-cann-path` (可选) 指定 `CANN` 包安装完后路径，不指定路径默认会从系统环境变量`ASCEND_TOOLKIT_HOME`中获取，如果该环境变量不存在则默认设置为 `/usr/local/Ascend/ascend-toolkit/latest`
  - `-o, –-output` (可选) 输出文件路径，默认为当前路径

