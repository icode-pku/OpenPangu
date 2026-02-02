
# msit benchmark功能使用指南

## 简介
本文介绍benchmark功能，用来针对指定的推理模型运行推理程序，并能够测试推理模型的性能（包括吞吐率、时延）。

## 工具安装
- 工具安装请见 [msit一体化工具使用指南](../install/README.md)
<br>如果在 `msit install benchmark` 的时候出现报错，显示需要 `use --no-check-certificate`，可通过`msit install benchmark --no-check`安装。
<br>但是，需要注意的是，--no-check参数，即--no-check-certificate会跳过检查目标网站的证书信息，有一定的安全风险，用户需要谨慎使用并自行承担后果。

## 使用方法
### 功能介绍
#### 使用入口
benchmark推理功能可以直接通过msit命令行形式启动模型测试。启动方式如下：
```bash
msit benchmark --om-model *.om
```
其中，*为通过ATC工具转换的OM离线模型文件名。

#### 安全风险提示

在模型文件传给工具加载前，用户需要确保传入文件是安全可信的，若模型文件来源官方有提供SHA256等校验值，用户必须要进行校验，以确保模型文件没有被篡改。

#### 参数说明
benchmark推理功能可以通过配置不同的参数，来应对各种测试场景以及实现其他辅助功能。

参数按照功能类别分为**基础功能参数**和**高级功能参数**：

- **基础功能参数**：主要包括输入文件及格式、debug、推理次数、预热次数、指定运行设备以及帮助信息等。
- **高级功能参数**：主要包括动态分档场景和动态Shape场景的benchmark推理测试参数以及profiler或dump数据获取等。

**说明**：以下参数中，参数和取值之间可以用“ ”空格分隔也可以用“=”等号分隔。例如：--debug 1或--debug=1。

##### 基础功能参数

| 参数名                     | 说明                                                                                                                                                                                     | 是否必选 |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| -------- |
| -om, --om-model         | 需要进行推理的OM离线模型文件。                                                                                                                                                                        | 是       |
| -i, --input              | 模型需要的输入。可指定输入文件所在目录或直接指定输入文件。支持输入文件格式为“NPY”、“BIN”。可输入多个文件或目录，文件或目录之间用“,”隔开。具体输入文件请根据模型要求准备。  若不配置该参数，会自动构造输入数据，输入数据类型由--pure-data-type参数决定。                                                  | 否       |
| -pdt, --pure-data-type   | 纯推理数据类型。取值为：“zero”、“random”，默认值为"zero"。 未配置模型输入文件时，工具自动构造输入数据。设置为zero时，构造全为0的纯推理数据；设置为random时，为每一个输入生成一组随机数据。                                                                                | 否       |
| -o, --output             | 推理结果保存目录。配置后会创建“日期+时间”的子目录，保存输出结果。如果指定--output-dirname参数，输出结果将保存到子目录output dirname下。不配置输出目录时，仅打印输出结果，不保存输出结果。                                                                                | 否       |
| -od, --output-dirname    | 推理结果保存子目录。设置该值时输出结果将保存到*output/output-dirname*目录下。  配合output参数使用，单独使用无效。 例如：--output */output* --output-dirname *output-dirname*                                                             | 否       |
| --outfmt            | 输出数据的格式。取值为：“NPY”、“BIN”、“TXT”，默认为”BIN“。  配合output参数使用，单独使用无效。 例如：--output */output* --outfmt NPY。                                                                                            | 否       |
| --debug                   | 调试开关。可打印model的desc信息和其他详细执行信息。1或true（开启）、0或false（关闭），默认关闭。                                                                                                                                   | 否       |
| -rm, --run-mode                | 推理执行前的数据加载方式：可取值：array（将数据转换成host侧的ndarray，再调用推理接口推理），files（将文件直接加载进device内，再调用推理接口推理），tensor（将数据加载进device内，再调用推理接口推理），full（将数据转换成host侧的ndarray，再将ndarray格式数据加载进device内，再调用推理接口推理），默认为array。 | 否 |
| -das, --display-all-summary     | 是否显示所有的汇总信息，包含h2d和d2h信息。1或true（开启）、0或false（关闭），默认关闭。                                                                                                                                        | 否       |
| --loop                    | 推理次数。默认值为1，取值范围为大于0的正整数。  profiler参数配置为true时，推荐配置为1。                                                                                                                                        | 否       |
| -wcount, --warmup-count  | 推理预热次数。默认值为1，取值范围为大于等于0的整数。配置为0则表示不预热。                                                                                                                                                     | 否       |
| -d, --device             | 指定运行设备。根据设备实际的Device ID指定，默认值为0。多Device场景下，可以同时指定多个Device进行推理测试，例如：--device 0,1,2,3。                                                                                                       | 否       |
| --divide-input            | 输入数据集切分开关，1或true（开启）、0或false（关闭），默认关闭。多Device场景下，打开时，工具会将数据集平分给这些Device进行推理。                                                                                                               | 否 |
| -h, --help                | 工具使用帮助信息。                                                                  | 否                                                                                                                                                                                            |


##### 高级功能参数

| 参数名                             | 说明                                                                                                                                                                                                                                                                                                                              | 是否必选 |
|---------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| -------- |
| -db, --dym-batch                  | 动态Batch参数，指定模型输入的实际Batch。 <br>如ATC模型转换时，设置--input_shape="data:-1,600,600,3;img_info:-1,3" --dynamic_batch_size="1,2,4,8"，dym-batch参数可设置为：--dym-batch 2。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 否       |
| -dhw, --dym-hw                    | 动态分辨率参数，指定模型输入的实际H、W。 <br>如ATC模型转换时，设置--input_shape="data:8,3,-1,-1;img_info:8,4,-1,-1" --dynamic_image_size="300,500;600,800"，dym-hw参数可设置为：--dym-hw 300,500。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 否       |
| -dd, --dym-dims                    | 动态维度参数，指定模型输入的实际Shape。 <br>如ATC模型转换时，设置 --input_shape="data:1,-1;img_info:1,-1" --dynamic_dims="224,224;600,600"，dym-dims参数可设置为：--dym-dims "data:1,600;img_info:1,600"。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 否       |
| -ds, --dym-shape                   | 动态Shape参数，指定模型输入的实际Shape。 <br>如ATC模型转换时，设置--input_shape_range="input1:\[8\~20,3,5,-1\];input2:\[5,3\~9,10,-1\]"，dym-shape参数可设置为：--dym-shape "input1:8,3,5,10;input2:5,3,10,10"。<br>动态Shape场景下，获取模型的输出size通常为0（即输出数据占内存大小未知），建议设置--output-size参数。<br/>例如：--dymShape "input1:8,3,5,10;input2:5,3,10,10" --output-size "10000,10000"                                                                                                                                                                                                                                                                                                                                                                         | 否       |
| -dr, --dym-shape-range              | 动态Shape的阈值范围。如果设置该参数，那么将根据参数中所有的Shape列表进行依次推理，得到汇总推理信息。<br/>对于单输入，配置格式为：name:1,3,200\~224,224-300或"name:1,3,200\~224,224-300"。例如：--dym-shape-range name:1,3,200\~224,224-300或--dym-shape-range "name:1,3,200\~224,224-300"。<br/>对于多输入，配置格式为"name1:1,3,200\~224,224-300;name2:1\~4,3,224,224"。例如：--dym-shape-range "name1:1,3,200\~224,224-300;name2:1\~4,3,224,224"。<br/>其中，name为模型输入名，“\~”表示范围，“-”表示某一位的取值。通过命令行配置动态Shape范围时，仅支持配置一组动态范围。若存在多个模型输入，各输入用英文分号;进行分隔。为确保参数的正确识别，请将整个配置字符串置于双引号内。<br/>当使用info文件进行配置时，可配置多组动态Shape范围。在*.info文件内，每组配置应单独放置在一行且无需使用引号。请注意，各配置行之间避免不必要的空行，以防止配置错误。info文件的格式如下：<br/>x1:1,3,32,32;x2:1\~4,3,32,32<br/>x1:1\~4,3,32,32;x2:1,3,32,32<br/>例如：--dym-shape-range *.info。 | 否       |
| -outsize, --output-size          | 指定模型的输出数据所占内存大小，多个输出时，需要为每个输出设置一个值，多个值之间用“,”隔开。<br>动态Shape场景下，获取模型的输出size通常为0（即输出数据占内存大小未知），需要根据输入的Shape，预估一个较合适的大小，配置输出数据占内存大小。<br>例如：--dym-shape "input1:8,3,5,10;input2:5,3,10,10" --output-size "10000,10000"                        | 否       |
| -asddm, --auto-set-dymdims-mode  | 自动设置动态Dims模式。1或true（开启）、0或false（关闭），默认关闭。<br/>针对动态档位Dims模型，根据输入的文件的信息，自动设置Shape参数，注意输入数据只能为npy文件，因为bin文件不能读取Shape信息。<br/>配合input参数使用，单独使用无效。<br/>例如：--input 1.npy --auto-set-dymdims-mode 1                                              | 否       |
| -asdsm, --auto-set-dymshape-mode  | 自动设置动态Shape模式。取值为：1或true（开启）、0或false（关闭），默认关闭。<br>针对动态Shape模型，根据输入的文件的信息，自动设置Shape参数，注意输入数据只能为npy文件，因为bin文件不能读取Shape信息。<br>配合input参数使用，单独使用无效。<br/>例如：--input 1.npy --auto-set-dymshape-mode 1                                           | 否       |
| -pf, --profiler                    | profiler开关。1或true（开启）、0或false（关闭），默认关闭。<br>profiler数据在--output参数指定的目录下的profiler文件夹内。配合--output参数使用，单独使用无效。不能与--dump同时开启。<br/>若环境配置了MSIT_NO_MSPROF_MODE=1，则使用--profiler参数采集性能数据时调用的是acl.json文件。                                           | 否       |
| --profiler-rename                       | 调用profiler落盘文件文件名修改开关，开启后落盘的文件名包含模型名称信息。1或true（开启）、0或false（关闭），默认开启。配合--profiler参数使用，单独使用无效。                                                                                                                                             |否|
| --dump                                  | dump开关。1或true（开启）、0或false（关闭），默认关闭。<br>dump数据在--output参数指定的目录下的dump文件夹内。配合--output参数使用，单独使用无效。不能与--profiler同时开启。                                                                                                                        | 否       |
| -acl, --acl-json-path               | acl.json文件路径，须指定一个有效的json文件。该文件内可配置profiler或者dump。当配置该参数时，--dump和--profiler参数无效。json文件中配置了profiler则默认解析成msprof命令执行profiling，若环境配置了MSIT_NO_MSPROF_MODE=1，则采用acl.json配置文件的方式采集性能数据。                                                      | 否       |
| --batch-size                            | 模型batchsize。不输入该值将自动推导。当前推理模块根据模型输入和文件输出自动进行组Batch。参数传递的batchszie有且只用于结果吞吐率计算。自动推导逻辑为尝试获取模型的batchsize时，首先获取第一个参数的最高维作为batchsize； 如果是动态Batch的话，更新为动态Batch的值；如果是动态dims和动态Shape更新为设置的第一个参数的最高维。如果自动推导逻辑不满足要求，请务必传入准确的batchsize值，以计算出正确的吞吐率。 | 否       |
| -oba, --output-batchsize-axis        | 输出tensor的batchsize轴，默认值为0。输出结果保存文件时，根据哪个轴进行切割推理结果，比如batchsize为2，表示2个输入文件组batch进行推理，那输出结果的batch维度是在哪个轴。默认为0轴，按照0轴进行切割为2份，但是部分模型的输出batch为1轴，所以要设置该值为1。 | 否       |
| -aipp, --aipp-config               | 带有动态aipp配置的om模型在推理前需要配置的AIPP具体参数，以.config文件路径形式传入。当om模型带有动态aipp配置时，此参数为必填参数；当om模型不带有动态aipp配置时，配置此参数不影响正常推理。 |否|
| --backend                               | 指定trtexec开关。需要指定为trtexec。配合--perf参数使用，单独使用无效。                                                           |否|
| --perf                                  | 调用trtexec开关。1或true（开启）、0或false（关闭），默认关闭。配合--backend参数使用，单独使用无效。                                         |否|
| -ec, --energy-consumption             | 能耗采集开关。1或true（开启）、0或false（关闭），默认关闭。需要配合--npu-id参数使用，默认npu id为0。                                         |否|
| --npu-id                                | 指定npu id，默认值为0。需要通过npu-smi info命令获取指定device所对应的npu id。配合--energy-consumption参数使用，单独使用无效。                |否|
| --pipeline                              | 指定pipeline开关，用于开启多线程推理功能。1或true（开启）、0或false（关闭），默认关闭。                                                   |否|
| --dump-npy                              | 指定dump-npy开关，用于开启dump结果自动转换功能。1或true（开启）、0或false（关闭），默认关闭。需要配合--output和--dump/--acl-json-path参数使用，单独使用无效。 |否|
| --threads                               | 指定threads开关，用于设置多计算线程推理时计算线程的数量。默认值为1，取值范围为大于0的正整数。需要配合--pipeline 1参数使用，单独使用无效。 |否|

### FAQ
使用过程中遇到问题可以参考[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)
### 使用场景

请移步[benchmark使用示例](../../examples/cli/benchmark/)
  | 使用示例               | 使用场景                                 |
  |-----------------------| ---------------------------------------- |
  |[01_basic_usage](../../examples/cli/benchmark/01_basic_usage)    | 基础示例，运行om模型的纯推理以及调试模式       |
  |[02_specify_input_data](../../examples/cli/benchmark/02_specify_input_data)|指定输入数据场景下的om模型推理|
  |[03_save_profiler_or_dump_data](../../examples/cli/benchmark/03_save_profiler_or_dump_data)|om模型推理过程中采集性能数据|
  |[04_save_output_data](../../examples/cli/benchmark/04_save_output_data)|om模型推理结束后保存输出结果|
  |[05_dynamic_grading](../../examples/cli/benchmark/05_dynamic_grading)|动态分档场景下的om模型推理|
  |[06_dynamic_shapes](../../examples/cli/benchmark/06_dynamic_shapes)|动态shape场景下的om模型推理|
  |[07_dynamic_aipp](../../examples/cli/benchmark/07_dynamic_aipp)|动态AIPP场景下的om模型推理|
  |[08_multi_device_scenario](../../examples/cli/benchmark/08_multi_device_scenario)|采用多个npu同步进行om模型的推理|
  |[09_trtexec](../../examples/cli/benchmark/09_trtexec)|集成NVIDIA trtexec工具进行onnx模型的推理|
  |[10_energy_consumption](../../examples/cli/benchmark/10_energy_consumption)|om模型推理获取功耗数据|
  |[11_multi_thread](../../examples/cli/benchmark/11_multi_thread)|多线程推理场景|
  |[12_dump_data_convert](../../examples/cli/benchmark/12_dump_data_convert)|dump数据自动转换|