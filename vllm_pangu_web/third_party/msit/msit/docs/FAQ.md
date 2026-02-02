- [FAQ](#faq)
  - [1. Import torch时报错: cannot allocate memory in static TLS block](#1-import-torch时报错-cannot-allocate-memory-in-static-tls-block)

# FAQ

## 1. Import torch时报错: cannot allocate memory in static TLS block
**报错提示** ImportError: {site-packages路径}/torch.libs/libgomp-6e1a1d1b.so.1.0.0: cannot allocate memory in static TLS block

**报错原因** 通常是由于线程局部存储（TLS）空间不足导致的，这种情况在使用某些库（如cv2、torch或libgomp）时较为常见。

**解决方案** 
1、使用LD_PRELOAD环境变量预加载相关库
```shell
# 找到这个文件位置
find / -name libgomp-6e1a1d1b.so.1.0.0
# 将文件路径添加到LD_PRELOAD环境变量中（具体路径根据上一步命令的回显决定, 下面路径仅供参考）
export LD_PRELOAD=$LD_PRELOAD:/root/anaconda3/envs/test/lib/python3.9/site-packages/torch.libs/libgomp-6e1a1d1b.so.1.0.0
```
2、升级glibc版本到2.32或更高, 以ubuntu系统为例：
```shell
ldd --version # 查看glibc版本
sudo apt-get update
sudo apt-get install libc6
```