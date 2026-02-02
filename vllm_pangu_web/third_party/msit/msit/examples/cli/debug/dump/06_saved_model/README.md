# Saved model dump

## 介绍
支持对使用Tensorflow框架保存下来的save_model格式模型进行tensor数据dump。

## 使用示例
- 1、saved_model npu dump  命令示例，**其中路径需使用绝对路径**
  ```sh
  msit debug dump -m /home/HwHiAiUser/prouce_data/resnet_offical_saved_model -dp npu
  -is "模型的输入shape" -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
  ```
  2、saved_model cpu dump  命令示例，**其中路径需使用绝对路径，cpu须在npu dump之后传入model文件夹下的ge_xxx.json文件作为--tf-json参数**
  ```sh
  msit debug dump -m /home/HwHiAiUser/prouce_data/resnet_offical_saved_model -dp cpu
  -is "模型的输入shape" -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test 
  --tf-json  /home/HwHiAiUser/result/test/{date}/model/ge_xxx.json
  ```
  - `-m, –-model` 指定原始saved_model模型路径
  - `-dp, --device-pattern` 指定dump的设备类型，支持cpu和npu，目前npu模式下只支持saved_model模型
  - `-is, --input-shape` (可选) 模型的输入shape
  - `-c，–-cann-path` (可选) 指定 `CANN` 包安装完后路径，不指定路径默认会从系统环境变量`ASCEND_TOOLKIT_HOME`中获取`CANN` 包路径，如果不存在则默认为 `/usr/local/Ascend/ascend-toolkit/latest`
  - `-o, –-output` (可选) 输出文件路径，默认为当前路径
  - `--tf-json`cpu dump必选参数，传入npu dump时生成的ge图相关json文件
  
