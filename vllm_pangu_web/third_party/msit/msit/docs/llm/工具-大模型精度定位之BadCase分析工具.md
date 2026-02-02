# msit Bad Case 分析工具
# 背景 
在大模型推理精度定位场景下，通常会出现 **两个模型** 在 **同一数据集下**，表现 **不一致** 的情况。比如，昇腾模型在 `NPU` 上，经过数据集评测，发现有两个问题出现回答 **错误** 的情况，但是同样的两个问题，其原生模型在 `GPU` 上的结果却是 **正确** 的，这时我们就称这两个问题为 `bad case`。

`msit` 工具针对这种场景，提供 **脚本** 和 **命令行** 两种交互方式来进行自动 `bad case` 分析，使能用户快速定位。后续衔接 `msit llm dump` 工具落盘模型推理数据，使用 `msit llm compare` 对比 logits，实现推理精度问题定位全流程。

相关内容：
- [精度问题定位全流程](大模型精度问题定位全流程.md)
- [如何识别 `bad case`](如何识别%20Bad%20Case.md)
- [如何找到问题 Token 落盘其 logits](加速库场景-输出Token的logits精度比对.md) [^1]
- [如何自动比对 logits](工具-大模型精度比对.md)

[^1]: `BoolQ` 数据集，由于其特性，如果出现精度问题，只会表现在第一个 `token` 上。对于代码生成模型在 `HumanEval` 上的评测，这个步骤十分有效。

# 介绍
进行 `bad case` 分析，需要两个步骤：

1.  **收集** ：首先需要收集比对的内容。比如要识别 `NPU` 相较于 `GPU` 场景的 `bad case`，首先需要针对 **同一个数据集** 进行模型推理，收集数据集问题，模型的回答，模型是否答对等必要项。
   
2.  **分析** ：当模型在数据集下的推理数据收集完毕之后，我们将两个不同场景下收集的结果进行分析。如，同一个问题，在 `NPU` 上记录为回答错误，但是在 `GPU` 上收集为回答正确，那么这个问题本身将会被当作为 `bad case` 进行输出。
   
`msit` 工具针对上述两个步骤，提供了两种使用方式：**脚本调用** 和 **命令行交互**

## 脚本调用
**脚本调用** 需要用户自行在推理脚本中调用我们提供的接口来进行统一的收集和后续的分析工作，针对的场景是用户自己手搓脚本执行模型的数据集推理。

### *class* msit_llm.Synthesizer(self, *, queries=None, input_token_ids=None, output_token_ids=None, passed=None)
`msit` 工具提供了 `Synthesizer` 接口使能用户完成 **收集** 工作，可以直接通过 `msit_llm` 包进行导入，以下是一个简单的使用示例：

```py
from msit_llm import Synthesizer

synthesizer = Synthesizer(
    queries='Question 1: Where is the capital of China?',
    input_token_ids=[123, 256, 123, 102, 132, 312, 515],
    output_token_ids=[37, 37, 123, 151, 625, 626, 745],
    passed='Correct'
)
synthesizer.to_csv()
```

`Synthesizer` 的构造函数提供四个参数用于收集必要信息
- `queries`: 为数据集的问题。通常数据集中的问题需要进行加工之后喂给模型，比如 `5-shots` 等。用户按照自己的需要提供相应的问题，之后的 `bad case` 筛选不会改变问题本身，传的是什么，输出的就是什么。
- `input_token_ids`: 为模型输入 token ids
- `output_token_ids`：为模型输出 token ids
- `passed`：模型结果相较于数据集标准答案是否正确。用户可以自己定义正确和错误的表现形式，如正确可以是 `True`，也可以是 `'Correct'` 等，但是需要注意的是，后续 `bad case` 分析比对的时候，如果出现格式、形式不一致的情况，会导致分析失败。因此，请务必确保输入的格式、形式是统一的。

> MindIE 模型推理脚本生成的 csv 文件的格式为 `'Correct'` 和 `'Wrong'`， 对应表示模型回答正确或者错误。
  
以下是一个简单示例：
```py
from msit_llm import Synthesizer

synthesizer = Synthesizer(
    queries='Question 1: Where is the capital of China?',
    input_token_ids=[123, 256, 123, 102, 132, 312, 515],
    output_token_ids=[37, 37, 123, 151, 625, 626, 745],
    passed=True
)
```

#### *method* from_args(self, *, queries=None, input_token_ids=None, output_token_ids=None, passed=None)
有时候，所有的信息无法同时获得。比如用户可能收集到了问题之后，暂时还获得不了模型的推理结果，常用的方案是维护一个数组，然后循环数据集结束之后，统一调用我们的接口函数进行收集。`Synthesizer` 针对这个问题提供了解决方案，用户可以使用 `from_args` 方法，可以逐个添加元素。对于暂未获取的元素，可以稍后添加，不需要额外开辟空间。

参数与构造函数保持一致，介绍请参考构造函数的介绍。

下列示例演示如何在添加一个 `queries` 到 `Synthesizer` 之后，再添加 `input_token_ids` [^2]：
```py
from msit_llm import Synthesizer

synthesizer = Synthesizer()

# Mimic dataset iteration
for ... in DataLoader(...):
    # suppose this is the query
    prompts = xxx
    
    # add 'query' to the synthesizer
    synthesizer.from_args(queries=prompts)

    # do something to the prompts
    
    # add 'input_token_ids' to the synthesizer
    input_token_ids = ...
    synthesizer.from_args(input_token_ids=input_token_ids)
```
[^2]：仅供演示，请以实际情况为主

#### *method* to_csv(self)
在使用 `from_args` 收集完毕之后，调用此方法将内容集成为 `csv` 文件。该文件会被放置于 `msit_bad_case/synthesizer` 文件夹下，文件命名规则为时间戳。

文件夹构造示例：
```sh
msit_bad_case/
└── synthesizer
    └── 20240802212356.csv
```

### *class* msit_llm.Analyzer
`msit` 工具提供 `Analyzer` 接口使能用户完成 **分析** 工作，可以直接通过 `msit_llm` 包进行导入，以下是一个简单的使用示例：

```py
from msit_llm import Synthesizer, Analyzer

# csv1 and csv2 are the results
Analyzer.analyze(csv1, csv2)

result_dict = {
    'queries': ['Q1', 'Q2'],
    'input_token_ids': [[1, 2, 3], [4, 5, 6]],
    'output_token_ids': [[7, 8, 9], [10, 11, 12]],
    'passed': [True, False]    
}
Analyzer.analyze(
    Synthesizer(**result_dict),
    csv_3 # csv results
)
```

#### *classmethod* analyze(golden, test)
当用户通过 `Synthesizer` 完成了 **收集** 工作，或者自行 **收集** 准备好了 `csv` 文件，下一步则是使用此方法进行 **分析** 。此方法会读取两个 `csv` 路径，检查路径是否合法，然后判断内容是否存在必须的表头 [^4]。随后根据两个 `csv` 文件的 `passed` 这一列，找到不一致的行，并返回行下的所有内容，输出 `csv` 文件。

如果比对发现没有不一致的行存在，则会提示用户，不会输出任何文件和文件夹。如果比对发现行数不一致，会通过打屏提醒用户，不一致的前五行是什么。

输出分析文件会存放于 `msit_bad_case/analyzer` 下，文件命名规则为时间戳

文件夹构造示例：
```sh
msit_bad_case/
└── analyzer
    └── 20240802212356.csv
```

使用示例 [^5]：
```py
from msit_llm import analyzer

# csv1 and csv2 are the results
Analyzer.from_csv(csv1, csv2)
```

[^4]: 必须的表头为，`queries`, `input_token_ids`, `output_token_ids`, `passed`

[^5]：仅供演示，请以实际 csv 路径为主

---

当用户通过 `Synthesizer` 做完了收集工作，并不需要输出 `csv` 文件之后再重新读入，多余的 IO 会影响性能。同样可以使用此方法，可以使能用户将 `Synthesizer` 的实例传入，省去冗余 IO 操作。两个参数既可以是 `csv` 文件路径，也可以是 `Sythesizer` 实例。

使用示例 [^6]：
```py
from msit_llm import analyzer

Analyzer.analyze(synthesizer1, synthesizer2)
Analyzer.analyze(synthesizer1, csv2)
Analyzer.analyze(csv1, synthesizer2)
```

[^4]: 必须的表头为，`queries`, `input_token_ids`, `output_token_ids`, `passed`

[^5]：仅供演示，请以实际 csv 路径为主

[^6]：仅供演示，请以实际场景为主

## 命令行交互
除了脚本使用之外，用户可以通过命令行的方式进行一键式进行 `bad case` 分析。用户可以传入收集的 `csv` 路径，可以进行一键式的 `bad case` 分析

参数介绍如下：

### Synopsis
```sh
msit llm analyze [OPTIONS...] <CSV_PATH>

OPTIONS:
    -g, --golden            标杆 csv 路径，必选
    -t, --test              测试 csv 路径，必选
    -h, --help              命令行参数帮助信息
    
...
    表示两个或多个，这里表示 '两个'

CSV_PATH:
    模型数据集推理结果：收集的 csv 存放路径
```

输出结果同 `Analyzer` 的 [输出一致](#classmethod-analyze-golden-test)。

### 使用示例
1. 两个都是 `csv`
```sh
msit llm analyze -g "msit_bad_case/synthesizer/20240802212356.csv" -t "msit_bad_case/synthesizer/202408022782516.csv"
```

# 示例
## 示例一
下列示例展示了如何在脚本中使用 `Analyzer` 和 `Synthesizer`：

```py
from msit_llm import Synthesizer, Analyzer


golden_synthesizer = Synthesizer(
   queries=['How are you?', 'Hello', 'What is your name?', 'What time is it?'],
   input_token_ids=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
   output_token_ids=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
   passed=['Correct', 'Wrong', 'Correct', 'Correct']
)

test_synthesizer = Synthesizer(
   queries=['Hello', 'How are you?', 'What is your name?', 'What time is it?', "Extra Question"],
   input_token_ids=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
   output_token_ids=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
   passed=['Wrong', 'Correct', 'Wrong', 'Correct']
)

Analyzer.analyze(golden=golden_synthesizer, test=test_synthesizer)
```

运行之后，出现下列日志信信息（日志时间，进程号以及文件名可能存在出入）
```sh
2024-09-25 13:49:59,699 - 243591 - msit_llm_logger - INFO - Checking if the header of csv is valid...
2024-09-25 13:49:59,706 - 243591 - msit_llm_logger - INFO - Checking if the header of csv is valid...
2024-09-25 13:49:59,706 - 243591 - msit_llm_logger - INFO - Analyzing...
2024-09-25 13:49:59,726 - 243591 - msit_llm_logger - INFO - 'Analyzer' has successfully finished the analysis, the result is stored at 'msit_bad_case/analyzer/20240925134959.csv'
```

查看 csv 文件，得到
```sh
$ cat msit_bad_case/analyzer/20240925134959.csv
queries,input_token_ids_golden,input_token_ids_test,output_token_ids_golden,output_token_ids_test,passed_golden,passed_test
What is your name?,"[12, 13, 14, 15, 16, 17]","[12, 13, 14, 15, 16, 17]","[12, 13, 14, 15, 16, 17]","[12, 13, 14, 15, 16, 17]",Correct,Wrong
```

## 示例二
下列示例展示了如何通过命令行进行 `bad case` 分析

```py
import time

from msit_llm import Synthesizer, Analyzer


golden_synthesizer = Synthesizer(
   queries=['Hello', 'How are you?', 'What is your name?', 'What time is it?', "Extra Question", "Extra Question"],
   input_token_ids=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
   output_token_ids=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
   passed=['Correct', 'Wrong', 'Correct', 'Correct']
)

test_synthesizer = Synthesizer(
   queries=['Hello', 'How are you?', 'What is your name?', 'What time is it?'], # 少几个问题
   input_token_ids=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22], [1], [2]],
   output_token_ids=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22], [2], [1]],
   passed=['Wrong', 'Correct', 'Wrong', 'Correct', 'Wrong', 'Wrong']
)

golden_synthesizer.to_csv()
time.sleep(1) # 由于示例速度过快，导致时间戳一样，出现覆写，故停一秒，实际场景不会出现这个问题
test_synthesizer.to_csv()
```

运行之后，出现下列日志信信息（日志时间，进程号以及文件名可能存在出入）
```sh
2024-09-25 13:55:09,487 - 243591 - msit_llm_logger - INFO - 'Sythesizer' has successfully finished the synthesis, the result is stored at 'msit_bad_case/synthesizer/20240925135509.csv'
2024-09-25 13:55:09,487 - 243591 - msit_llm_logger - INFO - 'Sythesizer' has successfully finished the synthesis, the result is stored at 'msit_bad_case/synthesizer/20240925135510.csv'
```

随后，使用命令行进行分析
```sh
msit llm analyze -g msit_bad_case/synthesizer/20240925135509.csv -t msit_bad_case/synthesizer/20240925135510.csv
```

出现打印日志如下（日志时间，进程号以及文件名可能存在出入）：
```sh
2024-09-25 13:55:09,487 - 243591 - msit_llm_logger - INFO - Checking if path '/root/msit_bad_case/synthesizer/20240925135509.csv' is valid...
2024-09-25 13:55:09,487 - 243591 - msit_llm_logger - INFO - Checking if path '/root/msit_bad_case/synthesizer/20240925135510.csv' is valid...
2024-09-25 13:55:09,487 - 243591 - msit_llm_logger - INFO - Checking if the header of csv is valid...
2024-09-25 13:55:09,487 - 243591 - msit_llm_logger - INFO - Checking if the header of csv is valid...
2024-09-25 13:55:09,487 - 243591 - msit_llm_logger - INFO - Analyzing...
2024-09-25 13:55:09,487 - 243591 - msit_llm_logger - INFO - 'Analyzer' has successfully finished the analysis, the result is stored at 'msit_bad_case/analyzer/20240925174505.csv'
2024-09-25 13:55:09,487 - 243591 - msit_llm_logger - WARNING - There are '2' quer(ies) not matched, below is a partial display of these unmatched queries:
        '4      Extra Question'
        '5      Extra Question'
```
注意，warning 出现的原因是由于 `golden` 比 `test` 多了两个问题，所以出现了问题 unmatched 的情况。如果这种现象很普遍，`bad case` 分析工具会打印前 `5` 个问题。
