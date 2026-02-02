# Dump data convert

## 介绍

自动将dump的结果bin文件转换为npy文件

## 运行示例

```bash
msit benchmark --om-model ./pth_resnet50_bs1.om --output ./output --dump 1 --dump-npy 1
```

输出结果示例如下：

```bash
    output/
    |-- 2023_01_03-06_35_53
    |-- 2023_01_03-06_35_53_summary.json
    |-- dump/
        |--20230103063551/
        |--20230103063551_npy/
```

在dump目录下除了原本的20230103063551子目录保存bin文件，还有转换后的20230103063551_npy子目录包含转换后的npy文件