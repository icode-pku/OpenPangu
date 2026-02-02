- [FAQ](#faq)
  - [1. 非root用户使用analyze工具时若使用root目录下/usr/local/Ascend/ascend-toolkit的文件，产生调用fast\_query shell失败的错误](#1-非root用户使用analyze工具时若使用root目录下usrlocalascendascend-toolkit的文件产生调用fast_query-shell失败的错误)
# FAQ

## 1. 非root用户使用analyze工具时若使用root目录下/usr/local/Ascend/ascend-toolkit的文件，产生调用fast_query shell失败的错误

- 错误信息：
```bash
2023-06-16 09:23:47,490 INFO : convert model to json, please wait...
2023-06-16 09:24:01,852 INFO : convert model to json finished.
2023-06-16 09:24:04,998 INFO : try to convert model to om, please wait...
2023-06-16 09:24:28,326 INFO : try to convert model to om finished.
2023-06-16 09:24:29,190 ERROR : load opp data failed, err:exec fast_query shell failed, err:2023-06-16 09:24:29 [ERROR] The input path may be insecure because it does not belong to you.
.
2023-06-16 09:24:29,247 INFO : analysis result has been written in out/result.csv.
2023-06-16 09:24:29,247 INFO : number of abnormal operators: 13.
2023-06-16 09:24:29,248 INFO : analyze model finished.

```
 

- 错误原因分析：

    当前analyze工具在检查模型支持度分析过程中会调用CANN包下算子速查工具进行检验，由于工具文件安全性检查要求调用算子速查工具脚本的使用者与该脚本的拥有者为同一人，故当非root用户使用root目录下/usr/local/Ascend/ascend-toolkit下文件时，将无法通过analyze工具的文件安全校验，所以无法调用。

- 解决方案：

    非root用户在/home/userxxx/目录下自行安装CANN开发者套件包，并正确配置相关环境变量（安装CANN包完成后根据提示），随后运行analyze工具即可。
