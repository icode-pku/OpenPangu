# One-Click Accuracy Analyzer (Inference)

### Overview
- This readme describes the **main.py** tool, or One-Click Accuracy Analyzer, for inference scenarios. This tool enables one-click network-wide accuracy analysis of TensorFlow and ONNX models. You only need to prepare the original model, offline model equivalent, and model input file. Beware that the offline model must be an .om model converted using the Ascend Tensor Compiler (ATC) tool, and the .bin input file must meet the input requirements of the model (multi-input models are supported).  
- For details about the usage restrictions of the tool, please visit: [CANN Community Edition/Restrictions (Only for Inference Scenarios)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/60RC1alphaX/developmenttools/devtool/atlasaccuracy_16_0035.html).

### Environment Setup
- Set up an operating and development environment powered by Ascend AI Processors including driver / firmware / CANN, referring [Ascend Documentation](https://www.hiascend.com/en/document)
- Install Python 3.7.5.
- Install benchmark tool, referring [ais_bench](https://gitcode.com/Ascend/msit/tree/master/msit/components/benchmark/README.md)
- **ONNX related python packages** `pip3 install onnxruntime onnx numpy`, If the installation of dependent modules fails using the pip command, it is recommended to execute the command **pip3 install --upgrade pip** to avoid installation failure due to low pip version.
- **TensorFlow related python packages**, referring [Centos7.6 installing tensorflow1.15.0](https://bbs.huaweicloud.com/blogs/181055).

### Usage
- Getting this package by downloading zip package or `git clone`:
  ```sh
  git clone https://gitcode.com/Ascend/msit.git
  ```
- Change directory to `compare`
  ```sh
  cd msit/msit/components/debug/compare
  ```
- Set CANN related environ variables. Change `/usr/local/Ascend/ascend-toolkit` to your own installed CANN path.
  ```sh
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```
- **Data prepare**
  - Path of the offline model (.om) adapted to Ascend AI Processor
  - Path of the original model (.pb or .onnx)
  - (Optional) Path of model input data (.bin)
- **Without model input specified**. **The pathes used here need to be absolute pathes**.
  ```sh
  python3 main.py -m /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
  ```
  - `-om, --offline-model-path` Path of the offline model (.om) adapted to the Ascend AI Processor.
  - `-m, --model-path` Path of the original model (.pb or .onnx). Currently, only .pb and .onnx models are supported.
  - `-c, --cann-path` (Optional) CANN installation path, defaulted to `/usr/local/Ascend/ascend-toolkit/latest`
  - `-o, --output-path` (Optional) Output path, defaulted to the current directory.
- **With model input specified**. **The pathes used here need to be absolute pathes**.
  ```sh
  python3 main.py -m /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -i /home/HwHiAiUser/result/test/input_0.bin -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
  ```
  - `-i, --input-path` Path of model input data, which is generated based on model inputs by default. Separate model inputs with commas (,), for example, `/home/input_0.bin, /home/input_1.bin`. This scenario will automatically perform group batch based on the file size and the actual input size of the model, but it is necessary to ensure that the shape of the data, except for the batch size, is consistent with the model input.

### Analysis Result Description
```sh
Used to distinguish between different actual inputs of models in dynamic shapes, but not in static shapes


{output_path}/{timestamp}/{input_name-input_shape}  # {input_name-input_shape} is used to record actual input shape for dynamic input, not exists if static
├-- dump_data
│   ├-- npu                          # npu dump data directory
│   │   ├-- {timestamp}              # model dump data directory, not exists if dump=False
│   │   │   └-- 0                    # Device ID
│   │   │       └-- {om_model_name}  # om model name
│   │   │           └-- 1            # model ID
│   │   │               ├-- 0        # Task ID， increase 1 each time a repeat task executed
│   │   │               │   ├-- Add.8.5.1682067845380164
│   │   │               │   ├-- ...
│   │   │               │   └-- Transpose.4.1682148295048447
│   │   │               └-- 1
│   │   │                   ├-- Add.11.4.1682148323212422
│   │   │                   ├-- ...
│   │   │                   └-- Transpose.4.1682148327390978
│   │   ├-- {time_stamp}
│   │   │   ├-- output_0.bin
│   │   │   └-- output_0.npy
│   │   └-- {time_stamp}_summary.json
│   └-- {onnx or tf}                         # onnx / TF model dump data directory
│       ├-- Add_100.0.1682148256368588.npy
│       ├-- ...
│       └-- Where_22.0.1682148253575249.npy
├-- input
│   └-- input_0.bin                          # random generated data if not specified input data
├-- model
│   ├-- {om_model_name}.json
│   └-- new_{om_model_name}.onnx             # onnx model using every node as output
├-- result_{timestamp}.csv                   # compare result file
└-- tmp                                      # tfdbg temp directory if -m {Tensorflow pb model}
```

### Comparison Result Analysis
- **The comparison result** is stored in `result_{timestamp}.csv`, The meaning of each field in the comparison result file is the same as that of the basic Model Accuracy Analyzer. For details, click the following link [Comparison Procedure (Inference Scenarios)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/60RC1alphaX/developmenttools/devtool/atlasaccuracy_16_0039.html)
- **analyser result** is printed at the end of execution, by analysing csv output line by line, and excluding NaN data. Then output the first monitor which value is not within the preset threshold.

  | Monitor                   | Threhold |
  | ------------------------- | -------- |
  | CosineSimilarity          | <0.99    |
  | RelativeEuclideanDistance | >0.05    |
  | KullbackLeiblerDivergence | >0.005   |
  | RootMeanSquareError       | >1.0     |
  | MeanRelativeError         | >1.0     |

  Printed output is in markdown table format.
  ```sh
  2023-04-19 13:54:10(1005)-[INFO]Operators may lead to inaccuracy:

  |                   Monitor |  Value | Index | OpType | NPUDump | GroundTruth |
  |--------------------------:|-------:|------:|-------:|--------:|------------:|
  |          CosineSimilarity | 0.6722 |   214 |    Mul |   Mul_6 |       Mul_6 |
  | RelativeEuclideanDistance |      1 |   214 |    Mul |   Mul_6 |       Mul_6 |
  ```
  **Python API usage**
  ```py
  from msquickcmp.net_compare import analyser
  _ = analyser.Analyser('result_2021211214657.csv')()
  # |                   Monitor |  Value | Index | OpType | NPUDump | GroundTruth |
  # |--------------------------:|-------:|------:|-------:|--------:|------------:|
  # |          CosineSimilarity | 0.6722 |   214 |    Mul |   Mul_6 |       Mul_6 |
  # | RelativeEuclideanDistance |      1 |   214 |    Mul |   Mul_6 |       Mul_6 |
  ```
  Two kind of strategy is defined.
  - `FIRST_INVALID_OVERALL` output only the first monitor which value is not within the preset threshold. Default value.
  - `FIRST_INVALID_EACH` output the first value not in the preset threhold for each monitor.
  ```py
  from msquickcmp.net_compare import analyser
  _ = analyser.Analyser('result_2021211214657.csv')(strategy=analyser.STRATEGIES.FIRST_INVALID_EACH)
  # |                   Monitor |   Value | Index |        OpType |                       NPUDump | GroundTruth # |
  # |--------------------------:|--------:|------:|--------------:|------------------------------:|------------:|
  # |          CosineSimilarity |  0.6722 |   214 |           Mul |                         Mul_6 |       Mul_6 |
  # | RelativeEuclideanDistance |       1 |   214 |           Mul |                         Mul_6 |       Mul_6 |
  # |       RootMeanSquareError |   4.061 |   241 |      ArgMaxV2 | PartitionedCall_ArgMax_118... | ArgMax_1180 |
  # |         MeanRelativeError |    2.99 |   241 |      ArgMaxV2 | PartitionedCall_ArgMax_118... | ArgMax_1180 |
  # | KullbackLeiblerDivergence | 0.01125 |   316 | BatchMatMulV2 |                    MatMul_179 |  MatMul_179 |
  ```

### Command-line Options

| Option&emsp                              | Description                              | Required |
| ---------------------------------------- | ---------------------------------------- | -------- |
| -m, --model-path                         | Path of the original model (.pb or .onnx). Currently, only .pb and .onnx models are supported. | Yes      |
| -om, --offline-model-path                | Path of the offline model (.om) adapted to the Ascend AI Processor. | Yes      |
| -i, --input-path                         | Path of model input data, which is generated based on model inputs by default. Separate model inputs with commas (,), for example, **/home/input\_0.bin, /home/input\_1.bin**. | No       |
| -c, --cann-path                          | CANN installation path, If no path is specified, it will be obtained from the system environment variable `ASCEND_TOOLKIT_HOME`. If the variable does not exist, the default path will be **/usr/local/Ascend/ascend-toolkit/latest**. | No       |
| -o, --output-path                        | Output path, defaulted to the current directory | No       |
| -s, --input-shape                         | Shape information of model inputs. Separate multiple nodes with semicolons, for example, **input_name1:1,224,224,3;input_name2:3,300**. By default, this option is left blank. **input_name** must be the node name in the network model before model conversion. | No       |
| -d, --device                              | Specify running device [0,255], default 0. | No       |
| --output-nodes                           | Output node specified by the user. Separate multiple nodes with semicolons, for example, **node_name1:0;node_name2:1;node_name3:0**. | No       |
| --output-size                            | Specify the output size of the model. If there are several outputs, set several values. In the dynamic shape scenario, the output size of the acquired model may be 0. The user needs to estimate a more appropriate value according to the input shape to apply for memory. Multiple output sizes are separated by English semicolons (,), such as "10000,10000,10000". | No       |
| --advisor           | Whether print advisor info on the end of execution | No    |
| -dr, --dymShape-range     | Dynamic shape range parameter. If this argument used, then all shapes list included in the argument will be considered into accuracy compare. (only support onnx model) <br/> For example:input_name1:1,3,200\~224,224-230;input_name2:1,300<br/> input_name must be the node name in the network model before model conversion; "\~" represents the range, a\~b\~c meaning [a: b :c]; "-" represents the exact value. | No  |
| --dump                   | Whether compare the accuracy of all the operation nodes output. Default True.(only support onnx model)<br/> For example: --dump False           | No  |
| --convert                 | Support om comparison result file data format from bin file to npy file, the generated npy file directory is ./dump_data/npu/{timestamp_bin2npy} folder. For example: --convert True | No    |

### Sample Execution
- Obtain the original model from [AIPainting_v2.pb](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/painting/AIPainting_v2.pb).
- Obtain the .om model from [AIPainting_v2.om](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/painting/AIPainting_v2.om).
- **Run the commands by referring to the above guide to execute the sample. If you want to try with model input data specified, run the command for the scenario where input data is not specified to generate input data files (.bin) as the input.**
