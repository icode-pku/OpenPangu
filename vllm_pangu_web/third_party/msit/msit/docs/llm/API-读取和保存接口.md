# 读取 dump 数据 API 使用指南
## API 简介提供
提供读取和保存 dump 数据的接口

### 导入依赖包
```python
from components.utils.tool import get_bin_data_from_dir, read_bin_data, convert_bin_data_to_pt, convert_bin_data_to_npy, save_torch_data, save_npy_data
```
- 按需导入

### API导览
|编号<td rowspan='1'>**主要文件**<td rowspan='1'>**接口分类**</td><td rowspan='1'>**接口名**</td>|
|----|
|1<td rowspan='7'>tool.py</td><td rowspan='1'>获取bin文件集合</td><td rowspan='1'>[get_bin_data_from_dir](#get_bin_data_from_dir)</td>|
|2<td rowspan='1'>读取bin文件</td><td rowspan='1'>[read_bin_data](#read_bin_data)</td>|
|3<td rowspan='1'>读取海思格式文件<td>[read_dump_data](#read_dump_data)</td>|
|4<td rowspan='2'>转换bin文件</td><td rowspan='1'>[convert_bin_data_to_pt](#convert_bin_data_to_pt)</td>|
|5<td rowspan='1'>[convert_bin_data_to_npy](#convert_bin_data_to_npy)</td>|
|6<td rowspan='2'>保存文件</td><td rowspan='1'>[save_torch_data](#save_torch_data)</td>|
|7<td rowspan='1'>[save_npy_data](#save_npy_data)</td>|

<a name="get_bin_data_from_dir"></a>

#### <font color=#DD4466>**get_bin_data_from_dir函数**</font>
**功能说明**

用于获取所有bin文件路径。

**函数原型**
```python
get_bin_data_from_dir(dump_data_dir, max_depth=20)
```
**参数说明**
|参数名|说明|是否必选|
|----|----|----|
|**dump_data_dir**| dump后存放所有bin数据的文件夹 |是|
|**max_depth**| 文件夹的深度 |否|

**返回值**

返回list:[str]类型文件夹路径集合。


<a name="read_bin_data"></a>

#### <font color=#DD4466>**read_bin_data函数**</font>
**功能说明**

读取bin文件

**函数原型**
```python
read_bin_data(bin_data_path)
```
**参数说明**
|参数名|说明|是否必选|
|----|----|----|
|**bin_data_path**| bin数据的文件路径 |是|

**返回值**

返回 TensorBinFile 类的中间数据格式。

<a name="read_dump_data"></a>

#### <font color=#DD4466>**read_dump_data函数**</font>
**功能说明**

读取海思格式的dump文件（主要用于debug dump组件和torch air的dump数据读取）

**函数原型**
```python
read_dump_data(dump_data_path)
```
**参数说明**
|参数名|说明|是否必选|
|----|----|----|
|**dump_data_path**| dump的海思数据的文件路径 |是|

**返回值**

返回包含算子输入和输出的bin格式文件，其中，输入和输出都为list类型。



<a name="convert_bin_data_to_pt"></a>

#### <font color=#DD4466>**convert_bin_data_to_pt函数**</font>
**功能说明**

将bin转换为tensor格式

**函数原型**
```python
convert_bin_data_to_pt(bin_tensor)
```
**参数说明**
|参数名|说明|是否必选|
|----|----|----|
|**bin_tensor**| TensorBinFile格式的数据 |是|

**返回值**

返回 Tensor 类型的数据格式。

- 一般想读取bin文件的数据并打印 tensor 格式可以使用如下示例的方式

```python
from components.utils.tool import read_bin_data, convert_bin_data_to_pt

bin_file_path = "/xxx/xxx/xx.bin"
data = convert_bin_data_to_pt(read_bin_data(bin_file_path))
print("bin data: ", data)
print("bin data shape: ", data.shape)
```

<a name="convert_bin_data_to_npy"></a>

#### <font color=#DD4466>**convert_bin_data_to_npy函数**</font>
**功能说明**

将bin转换为numpy格式

**函数原型**
```python
convert_bin_data_to_npy(bin_dump_data, dtype=DEFAULT_PARSE_DTYPE)
```
**参数说明**
|参数名|说明|是否必选|
|----|----|----|
|**bin_dump_data**| 内存中的海思格式dump数据 |是|
|**dtype**| 默认是uint8 |否|

**返回值**

返回两个列表，元素全为 Numpy 类型的算子输入参数list和算子输出参数list。


<a name="save_torch_data"></a>

#### <font color=#DD4466>**save_torch_data函数**</font>
**功能说明**

保存 torch 类型的数据

**函数原型**
```python
 save_torch_data(pt_data, pt_file_path)
```
**参数说明**
|参数名|说明|是否必选|
|----|----|----|
|**pt_data**| 想要保存的 tensor 数据 |是|
|**pt_file_path**| 保存的目录 |否|

**返回值**

无

<a name="save_npy_data"></a>

#### <font color=#DD4466>**save_npy_data函数**</font>
**功能说明**

保存 numpy 类型的数据

**函数原型**
```python
save_npy_data(npy_file_path, npy_data)
```
**参数说明**
|参数名|说明|是否必选|
|----|----|----|
|**npy_data**| 想要保存的 numpy 类型数据 |是|
|**npy_file_path**| 保存的目录 |否|

**返回值**

无