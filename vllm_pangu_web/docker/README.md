# 盘古模型推理-镜像与容器环境构建

## 一、从零构建所需image镜像

### 方式1（不推荐）

拉取vllm-ascend最新官方仓库代码构建镜像, 不推荐，镜像版本兼容性需要用户自行适配
```
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
docker build -t vllm-ascend-dev-image:latest -f ./Dockerfile .
```

### 方式2（推荐）
为方便使用，已适配好的v0.11.0版本的vllm-ascend仓库已集成到**vllm_pangu_web/third_party/vllm-ascend**路径

进入到**vllm_pangu_web/third_party/vllm-ascend**目录
确认Dockerfile文件内容：(你可以根据需求自定义拉取的镜像版本)
```
FROM quay.io/ascend/cann:8.3.rc1-910b-ubuntu22.04-py3.11
```
默认无需其他修改，直接运行：（**确保当前网络能够良好访问git！**）
```
docker build -t vllm-ascend-dev-image:latest -f ./Dockerfile .
```

默认镜像已适配的环境如下：
```
固件驱动>=23.0.6
CANN:8.3.rc1
VLLM:0.11.2+empty
VLLM-ASCEND:0.11.0rc1
torch:2.8.0+cpu
torch_npu:2.8.0
transformers:4.57.1
python3 3.11.13
```


## 二、构建docker容器

构建完镜像后，修改**vllm_pangu_web/docker/run_docker.sh**文件，并运行如下命令进行docker容器拉起:

```bash
bash run_docker.sh
```

### run_docker.sh说明

run_docker.sh中命令如下所示，你可以根据实际情况调整 `--device` 映射，当前映射了0,1,2,3四张卡。

```bash
docker run -it --net=host \
--name vllm-ascend-pangu \  # 定义自己的容器名字
--device /dev/davinci0 \    # 可根据实际情况进行调整，有多张卡就可以映射多个device
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /opt/pangu:/opt/pangu \  # 挂载的盘古模型所在的目录 需要替换为自己下载的盘古模型实际所在的目录
-v /data/home/user_path/workspace/vllm_pangu_web:/workspace/vllm_pangu_web \  # 挂载vllm_pangu_web项目目录，需要自定义挂载路径
e9133061a66c /bin/bash # 替换为自己实际构建的镜像id
```