# msit 工具安装

> 以下为最新的安装方式。7.0.0 版本之前安装是通过shell脚本安装，可以参考《[安装指南（历史版本）](./history.md)》

## 环境和依赖

msit推理工具的安装包括**msit包**和**依赖的组件包**的安装，其中依赖包可以根据需求只添加所需要的组件包。


|   依赖软件名称   | 是否必选 | 版本                                                                                                            | 备注                                                                                                                                                                                                                                                                                                                                                     |
|-----------------|---------|---------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CANN              | 必选  | 建议安装CANN商用版8.0.RC1以上版本                                                                                        | 请参见《[CANN-8.1.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)》安装昇腾设备开发或运行环境，即toolkit软件包。安装后请根据安装提示配置环境变量（可以参考  [配置环境变量](#安装前置说明) ）。                                                                                              |
| GCC               | 必选 | 7.3.0版本                                                                                                       | 请参见《[GCC安装指引](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/softwareinst/instg/instg_0123.html)》安装GCC编译器（centos 7.6平台默认为gcc 4.8编译器，可能**无法安装**本工具，建议更新gcc编译器后再安装）                                                                                                                                                                    |
| Python               | 必选 | 支持Python3.7.5+、Python3.8.x、Python3.9.x、Python3.10.x                                                           | 如需使用TensorFlow模型的精度对比功能，请安装Python3.7.5版本，其他功能安装Python3.7.0以上版本即可，需要注意Python与Torch的版本依赖关系（如Python3.8对应Torch2.1.0）                                                                                                                                                                                                                                       |
| TensorFlow  | 非必选 | 支持TensorFlow1.15.0（对应Python版本3.7.5~3.7.11）和TensorFlow2.6.5（对应Python版本3.7.5~3.7.11， 3.8.0~3.8.11， 3.9.0~3.9.2） | 参考 [Centos7.6上TensorFlow1.15.0 环境安装](https://bbs.huaweicloud.com/blogs/181055) 安装 TensorFlow1.15.0 环境。(如不使用TensorFlow模型的精度对比功能则不需要安装)                                                                                                                                                                                                                  |
| Caffe    | 非必选 | 与Python版本对应即可                                                                                                 | 参考 [Caffe Installation](http://caffe.berkeleyvision.org/installation.html) 安装 Caffe 环境。(如不使用 Caffe 模型的精度对比功能则不需要安装)                                                                                                                                                                                                                                    |
| Clang      | 非必选 | 与Python版本对应即可                                                                                                 | 依赖LLVM Clang，需安装[Clang工具](https://releases.llvm.org/)。                                                                                                                                                                                                                                                                                                 |
| ATB | 非必选 | 与MindIE-LLM版本配套即可                                                                             | 依赖加速库ATB，推荐使用MindIE镜像（包含ATB、MindIE-LLM、MindIE-Service等部件），镜像获取链接请参考[MindIE镜像](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)，安装所需版本MindIE镜像即可。也可以通过源码的方式编译安装，参考[MindIE-LLM](https://gitcode.com/Ascend/MindIE-LLM)说明安装加速库ATB工具, 注意需要申请加入Ascend组织后才可以查看该代码仓。(如不使用msit llm dump、opcheck、errcheck等功能则不需要安装)        |
| MindIE-LLM | 非必选 | 支持1.0.RC3和1.0.0，分别与CANN包版本8.0.RC1和8.0.RC2对应                                                                   | 依赖MindIE-LLM组件，推荐使用MindIE镜像（包含ATB、MindIE-LLM、MindIE-Service等部件），镜像获取链接请参考[MindIE镜像](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)，安装所需版本MindIE镜像即可。也可以通过源码的方式编译安装，参考[MindIE-LLM](https://gitcode.com/Ascend/MindIE-LLM)说明安装MindIE-LLM，注意需要申请加入Ascend组织后才可以查看该代码仓。(如不使用msit llm dump、opcheck、errcheck等功能则不需要安装) |

## msit安装
### 安装前置说明
- 安装开发运行环境的昇腾 AI 推理相关驱动、固件、CANN 包，参照 [CANN-8.1.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)。
- 安装后用户可通过 **设置CANN_PATH环境变量** ，指定安装的CANN版本路径，例如：export CANN_PATH=/xxx/Ascend/ascend-toolkit/latest。
- 若不设置，工具默认会从环境变量ASCEND_TOOLKIT_HOME和/usr/local/Ascend/ascend-toolkit/latest路径分别尝试获取CANN版本。


### 安装方式说明
安装方式包括：**源代码安装**和**pip源安装**，用户可以按需选取。
- [源代码安装](#源代码安装): 使用源码安装，保证是最新的 msit 功能。
- [pip源安装](#pip源安装): pip 安装 msit 包，一般一个季度发包一次。

常见报错可以参照[msit 安装常见问题](#常见问题-qa)

#### 源代码安装

```shell
git clone https://gitcode.com/Ascend/msit.git
# 1. git pull origin 更新最新代码 
cd msit/msit

# 2. 安装 msit 包
pip install .

# 3. 通过以下命令，查看组件名，根据业务需求安装相应的组件
# 参考各组件功能介绍:(https://gitcode.com/Ascend/msit/tree/master/msit#%E5%90%84%E7%BB%84%E4%BB%B6%E5%8A%9F%E8%83%BD%E4%BB%8B%E7%BB%8D)
msit install -h

# 4. 如果需要安装llm：
msit install llm

# 5. 安装之后可以使用 msit check 命令检查安装是否成功：
msit check llm 
```

**注意**：

- 通过 `msit install llm` 安装 llm 组件时，工具会尝试自动下载 ATB 预检功能（opcheck）所依赖的 nlohmann C++ json 库。下载过程中如遇到 `ERROR: cannot verify xxx.com's certificate` 证书不可信错误时，可采取以下处理方式：

  (1) 如果不使用 ATB 预检功能，则可忽略该报错。llm 组件的其它功能可正常使用。

  (2) 如果需使用 ATB 预检功能，则可手动下载 nlohmann json 库文件，而后通过 `--find-links` 参数指定库文件路径。例如，若库文件下载在 `/root/pkg` 目录下时，执行 `msit install llm --find-links=/root/pkg` 命令，进行 llm 组件安装。 nlohmann json 库的下载地址如下：

| 文件名 | 下载地址 | SHA256 Sum |
|------ | ------- | ---------- |
| v3.11.1.tar.gz | https://github.com/nlohmann/json/archive/refs/tags/v3.11.1.tar.gz | 598becb62ee0e01cf32795073c8ae09b6e95335cd43a4417b785d93ce105b0d0 |
| v3.11.2.tar.gz | https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz | d69f9deb6a75e2580465c6c4c5111b89c4dc2fa94e3a85fcd2ffcd9a143d9273 |
| v3.11.3.tar.gz | https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz | 0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406 |

#### pip源安装

```shell
# 1. 安装 msit 包
pip install msit

# 2. 通过以下命令，查看组件名，根据业务需求安装相应功能的组件
# 参考各组件功能介绍:(https://gitcode.com/Ascend/msit/tree/master/msit#%E5%90%84%E7%BB%84%E4%BB%B6%E5%8A%9F%E8%83%BD%E4%BB%8B%E7%BB%8D)
msit install -h

# 3. 如果需要安装llm：
msit install llm

# 4. 安装之后可以使用 msit check 命令检查安装是否成功：
msit check llm 
```

> Windows 下，仅支持安装 surgeon 组件

## 卸载

使用pip 卸载命令进行卸载。
1. 建议可以先看下目前安装的 msit 组件有哪些
```bash
# linux 
 pip list | grep -E "acl|msit|ais"
```
```shell
# Windows
pip list | findstr msit
```
结果参考：
```log
(base) PS C:\workspce\msit> pip list | findstr msit
msit-surgeon             7.0.0rc2
msit                  7.0.0rc2
```
2. 使用pip命令卸载您需要卸载的组件
```shell
# 仅仅卸载一个组件
pip uninstall msit-surgeon
# 卸载所有
pip uninstall msit-surgeon msit
```
> 如果您仅仅卸载某几个组件，请不要将 msit 卸载，会影响其他组件使用

> 注意：安装benchmark的时候，也会一并安装ais-bench和aclruntime。因此卸载时除了卸载msit benchmark，
> 还需要手动卸载ais-bench和aclruntime。

# 离线安装(仅支持linux)
因为部分用户可能需要在非联网机器中安装 msit， 下面提供离线安装指导：

### 下载所有软件包
该步骤需要在可以联网的机器中(需要保证两台机器是同样的系统，平台，Python 版本)，首先需要安装 msit ,然后使用 msit download 命令下载所依赖的包：

```bash
# 1. 首先安装 msit, 使用源码方式
git clone https://gitcode.com/Ascend/msit.git
cd msit/msit
pip install .

# 2. 再使用 msit download 命令，下载对应组件到目录：
# 2.1 仅仅下载某几个组件，例如下载 llm 到 ./pkg-cache 目录:
msit download llm --dest ./pkg-cache 
# 2.2 下载所有组件
msit download all --dest ./pkg-cache
```

### 离线机器安装
该步骤，首先将上一步骤中相关文件拷贝到离线机器，包括：msit 源码，下载的依赖包。开始安装

```bash 
# 1. 首先安装 msit
cd msit/msit
pip install .

# 2. 开始安装组件，如 llm :
msit install llm --find-links=./pkg-cache

# 3. 安装之后可以使用 msit check 命令检查安装是否成功：
msit check all 
```

**注意**：

- 离线安装不支持在 Windows 系统中安装。

- 离线安装首先要下载安装包，要保证下载安装包机器与离线安装机器一致。包括 Python 版本，平台，操作系统等。

- 离线安装不支持 llvm 的安装。


# 安装相关命令行参数

## 安装命令： msit install 

| 参数名              | 描述                                                                                                                                                                                            | 必选 |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ---- |
| comp_names       | 是一个位置参数，指定需要安装的组件，当前可以选择的有：`all`/`llm`/`surgeon`/`analyze`/`convert`/`profile`/`tensor-view`/`benchmark`/`compare`/`opcheck`/`graph`，具体选项可以使用`msit install --help` 查看。但指定为`all` 时，表明是需要安装所有组件 | 是   |
| --find-links, -f | 查找包的路径，一般用于离线安装时                                                                                                                                                                              | 否  |
| --no-check       | 安装benchmark功能，下载依赖包时，是否跳过检查目标网站的证书信息。仅在安装出现报错，显示需要 `use --no-check-certificate`时，使用--no-check参数。使用该参数后，会跳过检查目标网站的证书信息，有一定的安全风险，用户需要谨慎使用并自行承担后果。默认未配置，表示检查证书信息。                                | 否  |
| --help, -h       | 帮助信息                                                                                                                                                                                          | 否  |

```bash 
msit install llm
```

## 安装检查命令：msit check

| 参数名        | 描述                                                         | 必选 |
|------------| ------------------------------------------------------------ | ---- |
| comp_names | 是一个位置参数，指定需要检查的组件，当前可以选择的有：`all`/`llm`/`surgeon`/`analyze`/`convert`/`profile`/`tensor-view`/`benchmark`/`compare`/`opcheck`/`graph`，具体选项可以使用`msit check --help` 查看。但指定为`all` 时，表明是需要检查所有组件 | 是   |
| --help, -h | 帮助信息 | 否  |

```bash 
msit check all
```

## 附加构建命令：msit build-extra

部分组件在安装之后，会需要一些额外的构建动作，该步骤在 msit install 会自动执行。但是因为部分原因，比如前置包没有安装等，有可能构建失败。安装后可以使用 msit check 命令检查，会给出提示。用户可以使用 msit build-extra 重新构建。

| 参数名              | 描述     | 必选 |
|------------------|-----------------------| ---- |
| comp_names       | 是一个位置参数，指定需要构建的组件，当前可以选择的有：`llm`/`surgeon`/`analyze`/`convert`/`profile`/`tensor-view`/`benchmark`/`compare`/`opcheck`/`graph`，具体选项可以使用`msit build-extra --help` 查看 | 是   |
| --find-links, -f | 查找包的路径，一般用于离线安装时 | 否  |
| --help, -h       | 帮助信息 | 否  |

```bash 
msit build-extra llm
```

## 下载命令：msit download

下载安装包，用于离线安装场景

| 参数名        | 描述                                                         | 必选 |
|------------| ------------------------------------------------------------ | ---- |
| comp_names | 是一个位置参数，指定需要下载的组件，当前可以选择的有：`all`/`llm`/`surgeon`/`analyze`/`convert`/`profile`/`tensor-view`/`benchmark`/`compare`/`graph` ，具体选项可以使用`msit download --help` 查看| 是   |
| --dest     | 目标路径 | 否   |
| --help, -h | 帮助信息 | 否  |

```bash 
msit download llm
```

# 常见问题 Q&A

[参考：msit 安装常见问题](./FAQ.md ':include')

