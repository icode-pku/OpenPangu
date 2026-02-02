# Saved model compare

## 介绍
支持save_model格式的标杆模型和om模型进行精度比对。

## 使用示例
- 1、saved_model compare  命令示例，**其中路径需使用绝对路径**
  ```sh
  msit debug compare -gm /home/HwHiAiUser/prouce_data/resnet_offical_saved_model -om /home/HwHiAiUser/prouce_data/model/resnet50.om
  -saved_model_signature serving -saved_model_tag_set serve -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test -is "模型的输入shape"
  ```

  - `-gm, –-golden-model` 指定原始saved_model模型路径
  - `-om, –-om-model` 指定昇腾AI处理器的离线模型（.om）路径
  - `-saved_model_signature` (可选) tensorflow2.6框架下saved_model模型加载时需要的签名。                                                                  
  - `-saved_model_tag_set` (可选) tensorflow2.6框架下saved_model模型加载为session时的标签，可根据标签加载模型的不同部分。
  - `-c, –-cann-path` (可选) 指定 `CANN` 包安装完后路径，不指定路径默认会从系统环境变量`ASCEND_TOOLKIT_HOME`中获取`CANN` 包路径，如果不存在则默认为 `/usr/local/Ascend/ascend-toolkit/latest`
  - `-o, –-output` (可选) 输出文件路径，默认为当前路径
  - `-is, --input-shape` (可选) 模型的输入shape
