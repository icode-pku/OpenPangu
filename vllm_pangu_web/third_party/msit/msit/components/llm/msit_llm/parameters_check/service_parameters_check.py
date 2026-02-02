# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import json
import re

import pandas as pd
from tabulate import tabulate

from components.utils.constants import TEXT_FILE_MAX_SIZE, JSON_FILE_MAX_SIZE, LOG_FILE_MAX_SIZE, FileCheckConst
from components.utils.file_open_check import ms_open, sanitize_csv_value
from components.utils.file_utils import check_path_type, check_and_get_real_path
from msit_llm.common.log import logger

service_parameters = ['temperature', 'top_k', 'top_p', 'do_sample', 'seed', 'repetition_penalty', 'watermark',
                      'frequency_penalty', 'presence_penalty', 'length_penalty', 'ignore_eos']


def create_difference_dict(req_order, param, file1_value="N/A", file2_value="N/A"):
    if req_order == -1:
        return {"param": param, "file1_value": file1_value, "file2_value": file2_value}
    else:
        return {"req_order": req_order, "param": param, "file1_value": file1_value, "file2_value": file2_value}


def parse_parameter_line(line, params, current_request, file_line_number):
    extract_params = line.strip().split(':', 1)   # 防止value值中含冒号的情况
    if len(extract_params) == 2:
        key_part, value_part = extract_params   # 将参数分为key和value两部分，例如"temperature"和[0.69999]

        key = key_part.strip().strip('"')    # 去掉key中多余的空格和""
        value = value_part.strip().strip('[],')  # 去掉value中多余的空格和[]

        # ast模块无法解析true/false/null值, 先将其转成相对应的值
        value = value.replace('true', 'True').replace('false', 'False').replace('null', 'None')

        try:
            parsed_value = ast.literal_eval(value)
            if key == 'do_sample' and parsed_value is not None:
                parsed_value = bool(parsed_value)
            params[key] = parsed_value
        except (ValueError, SyntaxError) as e:
            logger.warning(f"Value parsing failed at line {file_line_number} (request {current_request}): {e}")
    elif '{' not in extract_params and '}' not in extract_params and extract_params != ['']:  # 过滤{ }和空行
        logger.warning(
            f"Unexpected parameter format at line {file_line_number}: "
            f"'{line.strip()}' (request: {current_request})"
        )
    else:
        pass


def extract_log_parameters(log_file_path):
    """
    Extract sampling parameters from the log file and return a dictionary.

    Parameters:
        log_file_path (str): Path to the log file.

    Returns:
        dict: A dictionary containing all the request parameters, structured as {request_id_counter: parameters_dict}.
    """
    result_dict = {}

    # 定义正则表达式
    start_pattern = re.compile(
        r'\[endpoint\] Sampling parameters for request id: (\S+)'
        r'|'
        r'Sampling parameters for trace ids \[([^]]*)\]:'
    )

    current_request = None
    brace_count = 0
    flag_begin = False
    match_cache = ""

    with ms_open(log_file_path, 'r', encoding='utf-8', max_size=LOG_FILE_MAX_SIZE) as f:
        request_id_counter = 1
        for file_line_number, line in enumerate(f, 1):
            # 匹配参数块开始行
            start_match = start_pattern.search(line)
            if start_match:
                # 处理未完成的前一个请求
                if current_request is not None:
                    logger.error(f"Abandon incomplete block for request {current_request}")

                # 提取请求标识符（request_id 或 trace_id）
                current_request = start_match.group(1) or start_match.group(2)
                brace_count = 0
                match_cache = start_match
                continue

            if match_cache and '{' in line:
                params = {}
                flag_begin = True
            # 处理参数块内容
            if flag_begin:
                brace_count += line.count('{')
                brace_count -= line.count('}')
                parse_parameter_line(line, params, current_request, file_line_number)
                if brace_count == 0:
                    result_dict[request_id_counter] = params
                    request_id_counter += 1
                    flag_begin = False
                    match_cache = ""
                    current_request = None
    return result_dict


def extract_txt_parameters(log_file_path):
    """
    Extract sampling parameters from the log file and return a dictionary.

    Parameters:
        log_file_path (str): Path to the log file.

    Returns:
        dict: A dictionary containing all the request parameters, structured as {request_id_counter: parameters_dict}.
    """
    with ms_open(log_file_path, 'r', max_size=TEXT_FILE_MAX_SIZE) as file:
        content = file.read()

    lines = content.split("\n")
    param_dict = {}
    request_id_counter = 1
    request_id_cache = []

    for line in lines:
        # 找到包含 `request_id: SamplingParams(...)` 的行
        if not line.strip() or ':' not in line:
            continue

        # 提取 `request_id` 和 `SamplingParams(...)` 部分
        try:
            request_id, param_str = line.split(":", 1)
        except ValueError as e:
            logger.warning(f"get param_str failed ,the line is {line}, fail reason is {e}")
            continue

        request_id = request_id.strip()
        param_str = param_str.strip()

        # 跳过重复request_id
        if request_id in request_id_cache:
            continue
        request_id_cache.append(request_id)

        # 利用正则提取SamplingParams后的参数
        start_match = re.findall(r'SamplingParams\(([^"]*)\)', param_str)
        # 检查是否包含SamplingParams(...)
        if not start_match:
            logger.warning('Please check whether the SamplingParams(...) is correct')
            continue
        params_str = start_match[0]

        # 使用栈结构解析参数并检查括号匹配
        params = []
        current_param = []
        stack = []
        error_flag = False
        for char in params_str:
            if char == ',' and not stack:
                params.append(''.join(current_param).strip())
                current_param = []
                continue
            current_param.append(char)
            if char in '([{':
                # 将对应的闭括号压入栈
                if char == '(':
                    stack.append(')')
                elif char == '[':
                    stack.append(']')
                elif char == '{':
                    stack.append('}')
            elif char in ')]}':
                if not stack:
                    logger.warning(f"Extra closing bracket '{char}' in line: {line}")
                    error_flag = True
                    break
                if char != stack[-1]:
                    logger.warning(f"Mismatched brackets: expected '{stack[-1]}', got '{char}' in line: {line}")
                    error_flag = True
                    break
                stack.pop()

        # 处理最后一个参数
        if not error_flag and current_param:
            params.append(''.join(current_param).strip())

        # 检查是否有未闭合的括号
        if not error_flag and stack:
            logger.warning(f"Unclosed brackets {stack} in line: {line}")
            error_flag = True

        if error_flag:
            logger.warning(f"Skipping invalid parameters in line: {line}")
            continue
        # 解析键值对并存入字典
        request_params = {}
        for param in params:
            if '=' not in param:
                logger.warning(f'Please check whether the parameter: {param} is correct.')
                continue
            key, value = param.split('=', 1)
            key = key.strip()
            value = value.strip()
            try:
                parsed_value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed_value = value.strip("'\"")  # 处理带引号的字符串
            request_params[key] = parsed_value

        if 'do_sample:True' in param_str:
            request_params['do_sample'] = True
        elif 'do_sample:False' in param_str:
            request_params['do_sample'] = False
        else:
            logger.warning('Please check that the content of the do_sample is correct.')

        param_dict[request_id_counter] = request_params  # 使用 request_id_counter 作为 param_dict 的键
        request_id_counter += 1  # 增加 request_id_counter
    return param_dict


def compare_parameters(dict_input1, dict_input2, path="", depth=0):
    if depth > 5:
        raise RecursionError(f"The recursion depth exceeds the maximum limit (5). Current path:{path}")
    """比对两个字典"""
    differences = []

    # 遍历第一个字典的key
    for key in dict_input1:
        new_path = f"{path}.{key}" if path else key
        if key not in dict_input2:
            differences.append(create_difference_dict(-1, new_path, dict_input1[key], "N/A"))
        else:
            # 如果值是字典，则递归调用
            if isinstance(dict_input1[key], dict) and isinstance(dict_input2[key], dict):
                differences.extend(compare_parameters(dict_input1[key], dict_input2[key], new_path, depth + 1))
            # 如果值是列表，则比对列表
            elif isinstance(dict_input1[key], list) and isinstance(dict_input2[key], list):
                # 比较数组的长度
                if len(dict_input1[key]) != len(dict_input2[key]):
                    differences.append(create_difference_dict(-1, new_path, dict_input1[key], dict_input2[key]))
                else:
                    for i, (item1, item2) in enumerate(zip(dict_input1[key], dict_input2[key])):
                        if item1 != item2:
                            differences.append(create_difference_dict(-1, f"{new_path}[{i}]",
                                                                      dict_input1[key][i], dict_input2[key][i]))
            else:
                # 比较值是否一致
                if dict_input1[key] != dict_input2[key]:
                    differences.append(create_difference_dict(-1, new_path, dict_input1[key], dict_input2[key]))

    # 遍历第二个字典的key，找出遗漏的key
    for key in dict_input2:
        new_path = f"{path}.{key}" if path else key
        if key not in dict_input1:
            differences.append(create_difference_dict(-1, new_path, "N/A", dict_input2[key]))
    return differences


def compare_service_parameters(dict_input1, dict_input2, req_order, differences):
    for param in service_parameters:
        if param not in dict_input1 and param not in dict_input2:
            continue
        elif param not in dict_input1:
            differences.append(create_difference_dict(req_order, param, "N/A", dict_input2[param]))

        elif param not in dict_input2:
            differences.append(create_difference_dict(req_order, param, dict_input1[param], "N/A"))
        else:
            if dict_input1[param] != dict_input2[param]:
                differences.append(create_difference_dict(req_order, param, dict_input1[param], dict_input2[param]))


def compare_and_generate(input1_params, input2_params):
    differences = []
    if len(input1_params) != len(input2_params):
        logger.error('Please make sure that the request parameters are the same on both sides')
    else:
        for req_order, params in input1_params.items():
            compare_service_parameters(input2_params[req_order], params, req_order, differences)
        generate_report(differences)


def service_params_check(input1, input2):
    check_path_type(input1, FileCheckConst.FILE)
    check_path_type(input2, FileCheckConst.FILE)
    input1 = check_and_get_real_path(input1, FileCheckConst.READ_ABLE)
    input2 = check_and_get_real_path(input2, FileCheckConst.READ_ABLE)

    if input1.endswith('.json') and input2.endswith('.json'):
        """加载两个 JSON 文件并进行比对"""
        with ms_open(input1, "r", encoding="utf-8", max_size=JSON_FILE_MAX_SIZE) as f_gpu, \
             ms_open(input2, "r", encoding="utf-8", max_size=JSON_FILE_MAX_SIZE) as f_npu:
            dict_input1 = json.load(f_gpu)
            dict_input2 = json.load(f_npu)
        differences = compare_parameters(dict_input1, dict_input2)
        generate_report(differences)
    elif input1.endswith('.txt') and input2.endswith('.log'):
        gpu_input = extract_txt_parameters(input1)
        npu_input = extract_log_parameters(input2)
        compare_and_generate(input1_params=npu_input, input2_params=gpu_input)
    elif input1.endswith('.log') and input2.endswith('.log'):
        npu_input1 = extract_log_parameters(input1)
        npu_input2 = extract_log_parameters(input2)
        compare_and_generate(npu_input1, npu_input2)
    else:
        logger.error('Please make sure that the type and format of the input file is correct')


def csv_input_safecheck(csv_input):
    logger.info('Safety check for csv content')
    for dif in csv_input:
        for key, value in dif.items():
            sanitize_csv_value(key)
            sanitize_csv_value(value)


def generate_report(differences, output_file="comparison_report.csv"):
    """生成比对报告并保存为CSV文件"""
    if not differences:
        logger.info("No differences found")
    else:
        csv_input_safecheck(differences)
        df = pd.DataFrame(differences)
        df['file1_value'] = df['file1_value'].astype(str)
        df['file2_value'] = df['file2_value'].astype(str)
        with ms_open(output_file, 'w') as f:
            df.to_csv(f, na_rep="None", index=False, encoding='utf-8')
        df = df.replace({"nan": "None"})
        table = tabulate(df, headers=df.columns, tablefmt="grid", missingval="None",
                         showindex=False, disable_numparse=True)
        logger.info(f"\n{table}")
        logger.info(f"Report saved to {output_file}")
