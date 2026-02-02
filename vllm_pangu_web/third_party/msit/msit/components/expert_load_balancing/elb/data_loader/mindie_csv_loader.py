# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import re
import json
from collections import defaultdict

import numpy as np

from components.expert_load_balancing.elb.data_loader.base_csv_loader import BaseCsvLoader
from components.utils.file_open_check import ms_open


TARGET_JSON_FILE_NAME = "model_gen_config.json"


class MindieCsvSumedLoader(BaseCsvLoader):
    def __init__(self, input_args):
        super().__init__(input_args)

    @staticmethod
    def check_input_path(input_path):
        """
        用于在工厂类中判断输入路径是否匹配当前loader，返回decode或prefill文件路径的list，不匹配返回空list
        """
        if not os.path.isdir(input_path):
            return None

        target_json_path = os.path.join(input_path, TARGET_JSON_FILE_NAME)
        if os.path.exists(target_json_path):
            return None

        target_decode_files = get_input_path(input_path, "decode")
        target_prefill_files = get_input_path(input_path, "prefill")
        if not target_decode_files and not target_prefill_files:
            return None

        res = {
            "decode": target_decode_files,
            "prefill": target_prefill_files
        }

        return res
    
    def load(self, target_files):
        target_decode_files = target_files.get("decode", None)
        target_prefill_files = target_files.get("prefill", None)

        
        decode_data = self.process_data(target_files=target_decode_files)
        prefill_data = self.process_data(target_files=target_prefill_files)

        res = {}
        if prefill_data is not None:
            res["prefill"] = prefill_data
            self.process_args(prefill_data)
            
        if decode_data is not None:
            res["decode"] = decode_data
            self.process_args(decode_data)

        if not res:
            raise ValueError("Loading data failed, cannot find decode data or prefill data in input path.")

        return res, self.input_args

    def process_data(self, target_files):
        if target_files:
            target_data = load_from_target_files(target_files)
            target_data = cut_to_same_len(target_data)
            target_data = np.concatenate(target_data, axis=1)
            # data shape is [layer_num, model_expert_num]
            return target_data
        else:
            return None

    def process_args(self, data):
        self.input_args.n_layers = data.shape[0]
        self.input_args.n_experts = data.shape[1]
        
        self.input_args.deploy_fp = self.input_args.output_dir
        self.input_args.n_nodes = self.input_args.num_nodes
        self.input_args.n_devices = self.input_args.num_npus
        self.input_args.redundant_experts = self.input_args.num_redundancy_expert


class MindieCsvSplitedLoader(BaseCsvLoader):
    def __init__(self, input_args):
        super().__init__(input_args)

    @staticmethod
    def check_input_path(input_path):
        if not os.path.isdir(input_path):
            return None
        
        target_json_path = os.path.join(input_path, TARGET_JSON_FILE_NAME)
        if not os.path.isfile(target_json_path):
            target_data_path = []
            sub_target_json_paths = []
            for sub_file in os.listdir(input_path):
                sub_path = os.path.join(input_path, sub_file)
                if not os.path.isdir(sub_path):
                    continue
                sub_json_path = os.path.join(sub_path, TARGET_JSON_FILE_NAME)
                if os.path.isfile(sub_json_path):
                    sub_target_json_paths.append(sub_json_path)
                    target_data_path.append(sub_path)
            if not sub_target_json_paths:
                return None
            target_json_path = sub_target_json_paths[0]
        else:
            target_data_path = [input_path]

        target_decode_files = get_nodes_file(target_data_path, "decode")
        target_prefill_files = get_nodes_file(target_data_path, "prefill")

        if not target_decode_files and not target_prefill_files:
            return None

        res = {
            "decode": target_decode_files,
            "prefill": target_prefill_files,
            "json": target_json_path
        }

        return res
    
    def load(self, target_files):
        target_decode_files = target_files.get("decode", None)
        target_prefill_files = target_files.get("prefill", None)
        target_json_file = target_files.get("json", None)
        self.process_args(target_json_file)

        res = {}
        decode_data = self.process_data(target_decode_files)
        prefill_data = self.process_data(target_prefill_files)
        if decode_data is not None:
            res["decode"] = decode_data
        if prefill_data is not None:
            res["prefill"] = prefill_data

        return res, self.input_args
        
    def process_data(self, target_files):
        if target_files:
            target_datas = []
            for rank in sorted(target_files.keys()):
                target_data_same_rank = load_from_target_files(target_files[rank])
                target_data_same_rank = cut_to_same_len(target_data_same_rank)
                target_data = sum(target_data_same_rank)
                target_datas.append(target_data)
            
            target_datas = cut_to_same_len(target_datas)
            return target_datas
        return None
        
    def process_args(self, target_json_file):
        with ms_open(target_json_file) as handle:
            config = json.load(handle)
        self.input_args.config_json = config


class MindieCsvSplitedLoaderWithTopK(MindieCsvSplitedLoader):
    def __init__(self, input_args):
        super().__init__(input_args)

    @staticmethod
    def check_input_path(input_path):
        if not os.path.isdir(input_path):
            return None
        res = MindieCsvSplitedLoader.check_input_path(input_path)
        if res is None:
            return None
        
        target_decode_files = res.get("decode", None)
        target_prefill_files = res.get("prefill", None)
        
        target_decode_topk_files = get_topk_file_from_file(target_decode_files, "decode")
        target_prefill_topk_files = get_topk_file_from_file(target_prefill_files, "prefill")

        if target_decode_topk_files is None and target_prefill_topk_files is None:
            return None
        
        res["decode_topk"] = target_decode_topk_files
        res["prefill_topk"] = target_prefill_topk_files

        return res

    def load(self, target_files):
        res, input_args = super().load(target_files)
        target_decode_topk_files = target_files.get("decode_topk", None)
        target_prefill_topk_files = target_files.get("prefill_topk", None)

        target_decode_topk_data = self.process_data(target_decode_topk_files)
        target_prefill_topk_data = self.process_data(target_prefill_topk_files)

        if target_decode_topk_data is not None:
            res["decode_topk"] = target_decode_topk_data
        if target_prefill_topk_data is not None:
            res["prefill_topk"] = target_prefill_topk_data
        
        return res, input_args

    def process_data(self, target_files):
        if target_files:
            target_datas = []
            for rank in sorted(target_files.keys()):
                target_data_same_rank = load_from_target_files(target_files[rank])
                target_data_same_rank = cut_to_same_len(target_data_same_rank)
                target_data = sum(target_data_same_rank)
                target_datas.append(target_data)
            
            target_datas = cut_to_same_len(target_datas)
            return target_datas
        return None


def get_bak_file(ori_files):
    res = {}
    for file in ori_files:
        bak_file = file.replace(".csv", "_bak.csv")
        res[file] = bak_file
    return res


def load_from_target_files(target_files):
    bak_target_files = get_bak_file(target_files)
    result_data = []
    for file in target_files:
        bak_file = bak_target_files[file]
        res = BaseCsvLoader.load_with_bak_file(file, bak_file)
        result_data.append(res)
    return result_data


def cut_to_same_len(data_list, length=None):
    length = min([data.shape[0] for data in data_list]) if length is None else length
    res = []
    for data in data_list:
        if data.shape[0] < length:
            raise ValueError("Loading data failed, shape of expert hot data < layer_num in model_gen_config.json.")
        res.append(data[:length, ])
    return res


def get_input_path(input_path, file_name_prefix):
    pattern = re.compile(rf"{file_name_prefix}_(\d+)\.csv")
    matched_list = []
    for f in os.listdir(input_path):
        file_path = os.path.join(input_path, f)
        if not os.path.isfile(file_path):
            continue
        matched = re.match(pattern, f)
        if not matched:
            continue
        matched_list.append((int(matched[1]), file_path))
    matched_list.sort()
    target_files = [item[1] for item in matched_list]
    return target_files


def file_matched(file_name, file_name_prefix):
    pattern = re.compile(rf"{file_name_prefix}_(\d+)\.csv")
    matched = re.match(pattern, file_name)
    if matched:
        return int(matched[1])
    return None


def get_nodes_file(target_data_paths, file_name_prefix):
    data_dict = defaultdict(list)
    for target_data_path in target_data_paths:
        for file_name in os.listdir(target_data_path):
            file_path = os.path.join(target_data_path, file_name)
            num_index = file_matched(file_name, file_name_prefix)
            if num_index is None:
                continue
            data_dict[num_index].append(file_path)
    return data_dict
            

def get_topk_file_from_file(target_files, prefix):
    if target_files is None:
        return None
    topk_files = defaultdict(list)
    for rank, file_list in target_files.items():
        for file in file_list:
            file_name = file.split(os.path.sep)[-1]
            topk_file = file.replace(file_name, file_name.replace(prefix, prefix + "_topk"))
            if os.path.isfile(topk_file):
                topk_files[rank].append(topk_file)
            else:
                return None
    return topk_files
