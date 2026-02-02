# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from components.expert_load_balancing.elb.data_loader.base_loader import DataType
from components.expert_load_balancing.elb.data_loader.mindie_csv_loader import \
    MindieCsvSumedLoader, MindieCsvSplitedLoader, MindieCsvSplitedLoaderWithTopK
from components.expert_load_balancing.elb.data_loader.vllm_pt_loader import VllmTensorLoader
from components.utils.security_check import get_valid_read_path


def get_loader_type(input_path):
    supported_loader_func_map = {
        DataType.MINDIE_SPLITED_CSV_WITH_TOPK: MindieCsvSplitedLoaderWithTopK.check_input_path,
        DataType.MINDIE_SPLITED_CSV: MindieCsvSplitedLoader.check_input_path,
        DataType.MINDIE_SUMED_CSV: MindieCsvSumedLoader.check_input_path,
        DataType.VLLM_SUMED_TENSOR: VllmTensorLoader.check_input_path
    }

    for data_type, func in supported_loader_func_map.items():
        target_files = func(input_path)
        if target_files is not None:
            return data_type, target_files

    return DataType.UNKNOWN_TYPE, None


class DataLoaderFactory:
    FactoryMap = {
        DataType.MINDIE_SUMED_CSV: MindieCsvSumedLoader,
        DataType.MINDIE_SPLITED_CSV: MindieCsvSplitedLoader,
        DataType.MINDIE_SPLITED_CSV_WITH_TOPK: MindieCsvSplitedLoaderWithTopK,
        DataType.VLLM_SUMED_TENSOR: VllmTensorLoader
    }

    @staticmethod
    def create_loader(input_path):
        data_type, target_files = get_loader_type(input_path)
        if data_type == DataType.UNKNOWN_TYPE:
            raise ValueError("cannot be unkown data type")
        return data_type, target_files
    

def load_data(args):
    data_type, files = DataLoaderFactory.create_loader(args.expert_popularity_csv_load_path)
    data_loader = DataLoaderFactory.FactoryMap[data_type]
    data_loader = data_loader(args)
    data, new_args = data_loader.load(files)
    new_args.data_type = data_type
    return data, new_args

