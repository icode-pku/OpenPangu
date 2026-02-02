# msit 工具安装



## 环境和依赖

msit推理工具的安装包括**msit包**和**依赖的组件包**的安装，其中依赖包可以根据需求只添加所需要的组件包。


|   依赖软件名称   | 是否必选 | 版本 |                      备注                      |
|-----------------|---------|------|------------------------------------------------|
| CANN              | 必 选  | 建议安装CANN商业版6.0.1以上版本 | 请参见《[CANN-6.0.1](https://www.hiascend.com/document/detail/zh/canncommercial/601/overview/index.html)》安装昇腾设备开发或运行环境，即toolkit软件包。安装后请根据安装提示配置环境变量（可以参考  [配置环境变量](https://www.hiascend.com/document/detail/zh/canncommercial/700/envdeployment/instg/instg_0040.html) ）。                       |
| GCC               | 必 选 | 7.3.0版本                                | 请参见《[GCC安装指引](https://www.hiascend.com/document/detail/zh/canncommercial/601/envdeployment/instg/instg_000086.html)》安装GCC编译器（centos 7.6平台默认为gcc 4.8编译器，可能**无法安装**本工具，建议更新gcc编译器后再安装） |
| Python               | 必选 | 支持Python3.7.5+、Python3.8.x、Python3.9.x、Python3.10.x | 如使用TensorFlow模型的精度比对功能则需要Python3.7.5版本                                                                                                               |
| TensorFlow  | 非必选 | -                                      | 参考 [Centos7.6上TensorFlow1.15.0 环境安装](https://bbs.huaweicloud.com/blogs/181055) 安装 TensorFlow1.15.0 环境。(如不使用TensorFlow模型的精度比对功能则不需要安装)                                                  |
| Caffe    | 非必选 | -    | 参考 [Caffe Installation](http://caffe.berkeleyvision.org/installation.html) 安装 Caffe 环境。(如不使用 Caffe 模型的精度比对功能则不需要安装)                                                                    |
| Clang      | 非必选 | -    | 依赖LLVM Clang，需安装[Clang工具](https://releases.llvm.org/)。                                                                                                    |


## msit安装

安装方式包括：**源代码一键式安装**和**按需手动安装不同组件**，用户可以按需选取。
- [源代码一键式安装](#源代码一键式安装): 一键式安装msit所有组件
- [按需手动安装不同组件](#按需手动安装不同组件): 可以按需选择所需msit组件，单个安装

常见报错可以参照[msit 安装常见问题](#常见问题-qa)

### 说明：
- 安装开发运行环境的昇腾 AI 推理相关驱动、固件、CANN 包，参照 [CANN-6.0.1](https://www.hiascend.com/document/detail/zh/canncommercial/601/overview/index.html)。安装后用户可通过 **设置CANN_PATH环境变量** ，指定安装的CANN版本路径，例如：export CANN_PATH=/xxx/Ascend/ascend-toolkit/latest。若不设置，工具默认会从环境变量ASCEND_TOOLKIT_HOME和/usr/local/Ascend/ascend-toolkit/latest路径分别尝试获取CANN版本。


### 源代码一键式安装

```shell
git clone https://gitcode.com/Ascend/msit.git
# 1. git pull origin 更新最新代码 
cd msit/msit

# 2. 添加执行权限
chmod u+x install.sh

# 3. 以下install.sh根据情况选一个执行
# a. 安装msit，包括debug、profile、benchmark、analyze等组
./install.sh
  
# b. 安装msit，包括debug、profile、benchmark、analyze等组件（安装clang等系统依赖库，需要提供sudo权限）
./install.sh --full
  
# c. 重新安装msit及其debug、profile、benchmark、analyze等组件
./install.sh --force-reinstall
```

### 按需手动安装不同组件

```shell
git clone https://gitcode.com/Ascend/msit.git
cd msit/msit

# 添加执行权限
chmod u+x install.sh

# 1. 只安装debug下面的surgeon组件
./install.sh --surgeon

# 2. 只安装debug下面的compare组件（由于依赖关系，默认安装benchmark组件）
./install.sh --compare

# 3. 只安装benchmark组件
./install.sh --benchmark

# 4. 只安装analyze组件
./install.sh --analyze

# 5. 只安装profile组件
./install.sh --profile

# 6. 只安装convert组件
./install.sh --convert
```


# 卸载
注：2023/08/01前下载的msit工具需要重新卸载再安装的msit以及各子工具
```shell
cd msit/msit

chmod u+x install.sh

# 1. 一个个询问式卸载
./install.sh --uninstall

# 2. 不询问式直接全部卸载
./install.sh --uninstall -y

# 3. 单独组件询问式卸载(例如surgeon组件)
./install.sh --uninstall --surgeon

# 4. 不询问式单独组件直接卸载(例如surgeon组件)
./install.sh --uninstall --surgeon -y
```

# windows 环境

windows 下，仅支持安装surgeon 组件

## 安装

```shell
git clone https://gitcode.com/Ascend/msit.git
cd msit/msit

# 1. 安装msit，包括surgeon组件
install.bat

# 2. 只安装debug下面的surgeon组件
install.bat --surgeon
```


## 卸载
注：2023/08/01前下载的msit工具需要重新卸载再安装的msit以及各子工具
```shell
cd msit/msit

# 1. 一个个询问式卸载
install.bat --uninstall

# 2. 不询问式直接全部卸载
install.bat --uninstall -y

# 3. 单独组件询问式卸载(例如surgeon组件)
install.bat --uninstall --surgeon

# 4. 不询问式单独组件直接卸载(例如surgeon组件)
install.bat --uninstall --surgeon -y
```

# 常见问题 Q&A

[参考：msit 安装常见问题](./FAQ.md ':include')

