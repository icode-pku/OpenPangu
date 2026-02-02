

- [1、Q： 安装失败，提示“find no cann path”，如何处理？](#1q-安装失败提示find-no-cann-path如何处理)
- [2、Q：使用./install.sh进行安装却报-bash: ./install.sh: Permission denied](#2q使用installsh进行安装却报-bash-installsh-permission-denied)
- [3、Q：常见报错 XXX requires YYY, which is not installed。](#3q常见报错-xxx-requires-yyy-which-is-not-installed)
- [4、Q：使用./install.sh，报错：/usr/bin/env: ‘bash\\r’: No such file or directory。](#4q使用installsh报错usrbinenv-bashr-no-such-file-or-directory)
- [5、Q：如何获取cann包路径？](#5q如何获取cann包路径)
- [6、Q: 之前安装msit能够使用，后续环境上的依赖包被其他人或者其他工具破坏了，使用msit时提示“pkg\_resources.VersionConflict:XXXXX”怎么办？](#6q-之前安装msit能够使用后续环境上的依赖包被其他人或者其他工具破坏了使用msit时提示pkg_resourcesversionconflictxxxxx怎么办)
- [7、Q：安装msit时，出现skl2onnx组件安装失败的情况](#7q安装msit时出现skl2onnx组件安装失败的情况)
- [8、Q: OpenSSL: error:1408F10B:SSL routines:ssl3\_get\_record:wrong version number](#8q-openssl-error1408f10bssl-routinesssl3_get_recordwrong-version-number)
- [9、Q：如果使用过程中出现`No module named 'acl'`，请检验CANN包环境变量是否正确](#9q如果使用过程中出现no-module-named-acl请检验cann包环境变量是否正确)
- [10、Q：如果安装过程中，出现以下提示：WARNING: env ACLTRANSFORMER\_HOME\_PATH is not set. Dump on demand package cannot be used.](#11q如果安装过程中出现以下提示warning-env-acltransformer_home_path-is-not-set-dump-on-demand-package-cannot-be-used)

## 1、Q： 安装失败，提示“find no cann path”，如何处理？

安装报错：

![输入图片说明](https://foruda.gitee.com/images/1686801650121824710/b64bf91e_9570626.png "屏幕截图")

**A：** 安装后用户可通过 设置CANN_PATH环境变量 ，指定安装的CANN版本路径，例如：export CANN_PATH=/xxx/Ascend/ascend-toolkit/latest/。若不设置，工具默认会从环境变量ASCEND_TOOLKIT_HOME和/usr/local/Ascend/ascend-toolkit/latest路径分别尝试获取CANN版本。

以下是设置CANN包环境变量的通用方法(假设CANN包安装目录为`ACTUAL_CANN_PATH`)：
* 执行如下命令：

    ```bash
    source $ACTUAL_CANN_PATH/Ascend/ascend-toolkit/set_env.sh
    ```

## 2、Q：使用./install.sh进行安装却报-bash: ./install.sh: Permission denied
**A：** 这是因为没有给install.sh添加执行权限导致的。

```
# 添加权限
chmod u+x install.sh

# 或使用
bash install.sh
```


## 3、Q：常见报错 XXX requires YYY, which is not installed。
![which is not installed](https://foruda.gitee.com/images/1686645293870003179/234cf67c_8913618.png "屏幕截图")
**A：** 这是由于本地安装包缺乏依赖导致的，并非msit报错，根据命令行提示安装即可。

```
pip3 install YYY
```

## 4、Q：使用./install.sh，报错：/usr/bin/env: ‘bash\r’: No such file or directory。 

![No such file or directory](./No_such_file.png "屏幕截图")

**A：** 这并不是文件报错，常见原因是因为代码在本地编译器中被默认更换了格式，在pycharm编辑器右下角将.sh文件格式由CRLF改为LF。
![CRLF改为LF](https://foruda.gitee.com/images/1686645370968699210/f44f04b3_8913618.png "屏幕截图")


## 5、Q：如何获取cann包路径？
**A：** 在这个命令中，export | grep ASCEND_HOME_PATH会将所有环境变量输出，并通过管道符将结果传递给grep命令。grep命令会查找包含ASCEND_HOME_PATH的行，并将结果传递给cut命令。cut命令会以等号为分隔符，提取第二个字段，即ASCEND_HOME_PATH的值，并将其输出。

```
echo $ASCEND_HOME_PATH
```

## 6、Q: 之前安装msit能够使用，后续环境上的依赖包被其他人或者其他工具破坏了，使用msit时提示“pkg_resources.VersionConflict:XXXXX”怎么办？

![输入图片说明](./VersionConflict.png "屏幕截图")

**A:** 说明msit的依赖包版本可能被升级到了不匹配版本，只需要重新安装下msit即可，即重新在msit/msit目录中，执行
```
./install.sh
```

或者执行
```
pip3 check
```
查看环境上的python组件存在哪些版本依赖不匹配，手动安装到对应版本即可，比如如下check结果表示protobuf版本不匹配，重新安装对应版本即可：

![输入图片说明](https://foruda.gitee.com/images/1686887221107606902/a0872e5b_9570626.png "屏幕截图")

执行
```
pip3 install protobuf==3.20.2
```

## 7、Q：安装msit时，出现skl2onnx组件安装失败的情况
![输入图片说明](https://foruda.gitee.com/images/1688461726292472393/721044b8_8277365.png "屏幕截图")
**A:** 
解决方法1：更换pip源，自行手动安装skl2onnx。执行
    ```
    pip3 install skl2onnx==1.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/  --trusted-host pypi.tuna.tsinghua.edu.cn
    ```

解决方法2：直接安装wheel包

下载[skl2onnx](https://pypi.tuna.tsinghua.edu.cn/packages/5e/59/0a47737c195da98d33f32073174b55ba4caca8b271fe85ec887463481f67/skl2onnx-1.14.1-py2.py3-none-any.whl)后，在下载好的目录中，执行
    ```
    pip3 install skl2onnx-1.14.1-py2.py3-none-any.whl
    ```

## 8、Q: OpenSSL: error:1408F10B:SSL routines:ssl3_get_record:wrong version number
**A:** 
解决方案：此问题为网络代理问题，一般配置代理为私人代理后重新安装msit即可，代理格式如下：
```
export http_proxy="http://用户名:密码@代理地址"
export https_proxy="http://用户名:密码@代理地址" 
```
注：密码要用url转义

## 9、Q：如果使用过程中出现`No module named 'acl'`，请检验CANN包环境变量是否正确
- **A：** 解决方案：
    > 以下是设置CANN包环境变量的通用方法(假设CANN包安装目录为`ACTUAL_CANN_PATH`)：
    >
    > * 执行如下命令：
    ```
    source $ACTUAL_CANN_PATH/Ascend/ascend-toolkit/set_env.sh
    ```
    > * 普通用户下`ACTUAL_CANN_PATH`一般为`$HOME`，root用户下一般为`/usr/local`

## 10、Q：如果安装过程中，出现以下提示：WARNING: env ACLTRANSFORMER_HOME_PATH is not set. Dump on demand package cannot be used.
**A:** 如果不使用大模型精度比对功能，忽略此告警。



  