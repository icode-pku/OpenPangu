# 安装

`pip3 install msguard`

# 使用

假设有如下函数

```py
def find_csv_path(input_dir):
	# 在 input_dir 下找到 csv 文件
	# xxx
```

需要对 `input_dir` 做校验，我们可以使用

## 装饰器用法
```py
from msguard import validate_params, Rule

@validate_params({'input_dir': Rule.input_dir_traverse})
def find_csv_path(input_dir):
	# xxx
```

这样，会在调用函数 `find_csv_path` 前，自动对入参 `input_dir` 进行路径校验。

注意事项：

1. `validate_params` 的入参是一个字典，字典的 key 值一定要和要校验的入参变量名一致，否则不会校验
2. 如果 `input_dir` 不满足条件会直接 raise 错误，如果不希望 raise 可以考虑下列其他使用方法

## 函数内部判断
```py
from msguard import validate_params, Rule

def find_csv_path(input_dir):
	if not Rule.input_dir_traverse.is_satisfied_by(input_dir):
        return
	# xxx
```

`Rule` 承载了所有的常规路径校验，其每一个属性都是一个校验项 `Constraint`，每个校验项会有一个方法叫做 `is_satisfied_by`，用于判断入参是否满足校验。因此 `Rule.input_dir_traverse.is_satisfied_by(input_dir)` 判断 `input_dir` 是否符合 `input_dir_traverse` 的要求。这里的逻辑是，如果不符合，则 `return`。这样，就可以避免直接 raise 导致的程序中断

## argparse 用法
```py
from msguard import validate_args

parser.add_argument('--input-path', type=validate_args(Rule.input_dir_traverse), help="输入目录")
```

我们很多命令行用法都使用的是标准库 `argparse`，该库的 `add_argument` 函数支持一个入参 `type`，可以支持自定义函数校验命令行入参。
安全库提供 `validate_args` 用法，接受一个入参 Constraint，用来限制命令行输入，如果判断通过，则返回输入的路径。在最外层进行安全防护，并传递真实路径杜绝软链接风险。

## 任意函数包裹
```py
import os

import pandas
from msguard import validate_params, Rule


def read_csv_from_dir(input_dir):
    csv_file = os.path.join(input_dir, "a.csv")
    df = validate_params({"filepath_or_buffer": Rule.input_file_read})(pd.read_csv)(csv_file)
```

我们有很多时候会遇到入参不是需要校验的对象，拼接之后的路径需要被校验。除了使用 [装饰器](#装饰器用法) 的方式外，我们可以将任意函数显式地包裹，如果不符合要求则自动报错。
这里我们通过 `validate_params` 装饰了 `pd.read_csv` 的三方库函数，要求它的入参 `"filepath_or_buffer"` 必须满足 `Rule.input_file_read`，否则报错。但是需要注意，这里 `"filepath_or_buffer"` 和 `pd.read_csv` 的第一个入参的变量名是一致的，如果入参名改动，则校验无效。

## 其他场景

除此之外，安全库还包含了其他的常用场景，如
- `open_s`：原 `msopen`，会在创建文件，读取文件时自动进行校验，返回句柄
- `walk_s`：会在遍历目录的时候自动进行最大文件数和深度判断
- `update_env_s`：会在添加环境变量搜索路径的时候，自动判断是否为绝对路径，不会添加 trailing ":"。防护动态库劫持。

还有其他的用法等待大家的探索
