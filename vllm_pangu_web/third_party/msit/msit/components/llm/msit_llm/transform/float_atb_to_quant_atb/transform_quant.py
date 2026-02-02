# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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

import os
from msit_llm.transform.float_atb_to_quant_atb.utils import (
    check_libclang_so,
    filter_chinese_char,
    get_args_and_options,
    print_spelling,
    print_update_info,
    update_contents,
)
from msit_llm.common.log import logger
from msit_llm.common.utils import load_file_to_read_common_check
from components.utils.file_open_check import ms_open
from components.utils.constants import TENSOR_MAX_SIZE

USING_SCALE_BIAS_ITEMS = ["IN_QKV", "IN_QMIX", "IN_KMIX", "IN_VMIX", "IN_SELFOUTLINEAR", "IN_MLP"]
USING_SPARSE_INDEX_ITEMS = ["IN_QKV", "IN_QMIX", "IN_KMIX", "IN_VMIX"]
INTERMIDATE_PREFIX = "INTERMIDATE_"
CPP_TEMP_FILE_NAME = "transform_quant_cpp_temp.cpp"
HPP_TEMP_FILE_NAME = "transform_quant_cpp_temp.hpp"
DEQSCALE_SUFFIX = "DEQSCALE"
BIAS_SUFFIX = "BIAS"
INDEX_SUFFIX = "INDEX"
WEIGHT_SUFFIX = "WEIGHT"
IN_BETA = "IN_BETA"
IN_HOLDER = "IN_HOLDER"
INS = "// Quant weights\n"


class TransformQuant:
    def __init__(self, cpp_file_path, hpp_file_path=None, indent=4, enable_sparse=False):
        from clang import cindex

        if hpp_file_path is None:
            base_name = os.path.splitext(cpp_file_path)
            hpp_file_path = (base_name + ".h") if os.path.exists(base_name + ".h") else (base_name + ".hpp")

        cpp_cursor, cpp_contents = self.parse_file_as_cursor(cpp_file_path)
        if not list(cpp_cursor.get_children()):
            raise RuntimeError('AST has no child nodes')
        cpp_children = list(next(list(cpp_cursor.get_children())[-1].get_children()).get_children())
        self.cpp_contents, self.cpp_children = cpp_contents, cpp_children
        self.cpp_file_path, self.hpp_file_path = cpp_file_path, hpp_file_path
        self.enable_sparse = enable_sparse

        if os.path.exists(hpp_file_path):
            hpp_cursor, hpp_contents = self.parse_file_as_cursor(hpp_file_path)
            hpp_children = list(next(list(hpp_cursor.get_children())[-1].get_children()).get_children())
            self.hpp_contents, self.hpp_children = hpp_contents, hpp_children
        else:
            self.hpp_contents, self.hpp_children = "", []

        # Check Enum item in cpp first, if not found, try hpp file then
        in_tensor_added = None
        for cur_cursor in cpp_children:
            if cur_cursor.kind == cindex.CursorKind.ENUM_DECL:
                in_tensor_added = self.get_in_tensor_added(cur_cursor)

        if in_tensor_added is None:
            for cur_cursor in hpp_children:
                if cur_cursor.kind == cindex.CursorKind.ENUM_DECL:
                    in_tensor_added = self.get_in_tensor_added(cur_cursor)

        if in_tensor_added is None:
            message = "Fount none enum class in both cpp and h file, `in_tensor_added` is empty"
            logger.error(message)
            raise ValueError(message)
        self.in_tensor_added, self.indent, self.indent_prefix = in_tensor_added, indent, " " * indent
        self.cursor_kind = cindex.CursorKind

    def __call__(self):
        cpp_contents = self.do_transform_quant(is_cpp=True)
        target_cpp_file_path = self.to_quant_file_path(self.cpp_file_path, enable_sparse=self.enable_sparse)
        self.write_to_file(target_cpp_file_path, cpp_contents)

        hpp_contents = self.do_transform_quant(is_cpp=False)
        target_hpp_file_path = self.to_quant_file_path(self.hpp_file_path, enable_sparse=self.enable_sparse)
        self.write_to_file(target_hpp_file_path, hpp_contents)
        return target_cpp_file_path, target_hpp_file_path

    @staticmethod
    def parse_file_as_cursor(file_path):
        from clang import cindex

        file_ext = os.path.splitext(file_path)[-1]
        temp_file = CPP_TEMP_FILE_NAME if file_ext in [".c", ".cpp"] else HPP_TEMP_FILE_NAME

        file_path = load_file_to_read_common_check(file_path)

        contents = ms_open(file_path, max_size=TENSOR_MAX_SIZE).read()
        contents = filter_chinese_char(contents)
        args, options = get_args_and_options()
        parser = cindex.Index.create(excludeDecls=True)
        tu = parser.parse(temp_file, args=args, unsaved_files=[(temp_file, contents)])
        return tu.cursor, contents

    @staticmethod
    def to_quant_file_path(file_path, enable_sparse=False):
        prefix = "sparse_quant_" if enable_sparse else "quant_"
        return os.path.join(os.path.dirname(file_path), prefix + os.path.basename(file_path))

    @staticmethod
    def write_to_file(file_path, contents):
        with ms_open(file_path, "w") as ff:
            ff.write(contents)

    @staticmethod
    def is_in_tensor_count(cur_spelling):
        return cur_spelling.startswith("IN_") and cur_spelling.endswith("_COUNT")

    def get_in_tensor_added(self, enum_cursor):
        added_items = []
        all_enums, op_weight_with_bias = [], {}
        for enum_item in enum_cursor.get_children():
            enum_item_spelling = enum_item.spelling
            all_enums.append(enum_item_spelling)
            if not enum_item_spelling.endswith(BIAS_SUFFIX):
                continue
            enum_op_name = enum_item_spelling[: -len(BIAS_SUFFIX)]
            enum_op_name = enum_op_name[:-1] if enum_op_name.endswith("_") else enum_op_name
            if enum_op_name + WEIGHT_SUFFIX in all_enums:
                op_weight_with_bias[enum_op_name + WEIGHT_SUFFIX] = enum_item_spelling
            elif enum_op_name + "_" + WEIGHT_SUFFIX in all_enums:
                op_weight_with_bias[enum_op_name + "_" + WEIGHT_SUFFIX] = enum_item_spelling

        for enum_item in enum_cursor.get_children():
            enum_item_spelling = enum_item.spelling
            if enum_item_spelling.endswith(BIAS_SUFFIX):
                continue
            if any([ii in enum_item_spelling for ii in USING_SCALE_BIAS_ITEMS]):
                added_items.append(enum_item_spelling + "_" + DEQSCALE_SUFFIX)
                if enum_item_spelling not in op_weight_with_bias:
                    added_items.append(enum_item_spelling + "_" + BIAS_SUFFIX)
            # For transforming to sparse-quant model
            if self.enable_sparse and any([ii in enum_item_spelling for ii in USING_SPARSE_INDEX_ITEMS]):
                added_items.append(enum_item_spelling + "_" + INDEX_SUFFIX)

        if IN_BETA not in all_enums:
            added_items.append(IN_BETA)
        return added_items

    def update_in_tensor_added_enum(self, enum_cursor, contents):
        insert_position = enum_cursor.extent.end.offset - 1
        for enum_item in enum_cursor.get_children():
            if enum_item.spelling.startswith(INTERMIDATE_PREFIX):
                insert_position = contents[: enum_item.extent.start.offset].rfind("\n") + 1
                break

        insert_contents = "".join([self.indent_prefix + ii + ",\n" for ii in self.in_tensor_added])
        insert_contents = "\n" + self.indent_prefix + INS + insert_contents + "\n"
        return insert_contents, insert_position, insert_position

    def update_in_tensor_count(self, cursor, contents):
        # Extract the substring starting from the cursor's end offset
        cur_contents = contents[cursor.extent.end.offset:]
        next_semicolon_pos = cur_contents.find(";")
        if next_semicolon_pos == -1:
            raise ValueError("Semicolon not found; cannot parse in_tensor_count.")

        # Extract the value after the last '=' and strip whitespace
        raw_value = cur_contents[:next_semicolon_pos].split("=")[-1].strip()
        try:
            cur_in_tensor_count = int(raw_value)
        except ValueError as e:
            raise ValueError(f"Failed to convert '{raw_value}' to an integer.") from e

        # Increase the tensor count by the number of added tensors
        cur_in_tensor_count += len(self.in_tensor_added)
        insert_contents = " = {}".format(cur_in_tensor_count)
        insert_start = cursor.extent.end.offset
        insert_end = insert_start + next_semicolon_pos
        return insert_contents, insert_start, insert_end

    def update_from_json(self, cursor, contents):
        if len(list(cursor.get_arguments())) < 2:
            raise RuntimeError('The number of arguments for AST must be greater than 2, '
                               f'but got {len(list(cursor.get_arguments()))}')
        json_param, in_param = list(cursor.get_arguments())[:2]
        json_param_spelling, in_param_spelling = json_param.spelling, in_param.spelling
        insert_position = cursor.extent.end.offset - 1

        param_format = "    " + json_param_spelling + '.at("{}").get_to(' + in_param_spelling + ".{});"
        in_tensor_added_lower = [ii.lower() for ii in self.in_tensor_added if not ii.endswith(INDEX_SUFFIX)]
        insert_contents = "\n".join([param_format.format(ii, ii) for ii in in_tensor_added_lower]) + "\n"
        return insert_contents, insert_position, insert_position

    def update_param_struct(self, cursor, contents):
        insert_start = insert_end = cursor.extent.end.offset - 1
        insert_contents = "\n"
        for in_tensor in self.in_tensor_added:
            if in_tensor.endswith(DEQSCALE_SUFFIX):
                insert_contents += f"{self.indent_prefix}float {in_tensor.lower()} = 1;\n"
            elif in_tensor.endswith(BIAS_SUFFIX):
                insert_contents += f"{self.indent_prefix}int {in_tensor.lower()} = 0;\n"
        return insert_contents, insert_start, insert_end

    def is_layer_function(self, cursor):
        # This check may change depending on actual situation
        if cursor.kind != self.cursor_kind.FUNCTION_DECL:
            return False
        if not cursor.is_definition():
            return False
        if len(list(cursor.get_arguments())) < 2:
            return False
        op_parameter = list(cursor.get_arguments())[1]
        parameter_type = "".join([ii.spelling for ii in op_parameter.get_tokens()][:3])
        return parameter_type == "atb::Operation"

    def do_transform_quant(self, is_cpp=True):
        updates = []
        if is_cpp:
            children, contents = self.cpp_children, self.cpp_contents
        else:
            children, contents = self.hpp_children, self.hpp_contents

        print_spelling(children, info="Children parts from cpp file: ", level="info")
        for cur_cursor in children:
            cur_spelling = cur_cursor.spelling
            print_spelling(cur_cursor, info=f"current cursor: {cur_spelling}, {cur_cursor.kind}, ")
            if cur_cursor.kind == self.cursor_kind.ENUM_DECL:
                insert_contents, insert_start, insert_end = self.update_in_tensor_added_enum(cur_cursor, contents)
            elif cur_cursor.kind == self.cursor_kind.STRUCT_DECL:
                insert_contents, insert_start, insert_end = self.update_param_struct(cur_cursor, contents)
            elif cur_cursor.kind == self.cursor_kind.VAR_DECL and self.is_in_tensor_count(cur_spelling):
                insert_contents, insert_start, insert_end = self.update_in_tensor_count(cur_cursor, contents)
            elif (
                cur_cursor.kind == self.cursor_kind.FUNCTION_DECL
                and cur_cursor.is_definition()
                and cur_spelling == "from_json"
            ):
                insert_contents, insert_start, insert_end = self.update_from_json(cur_cursor, contents)
            elif self.is_layer_function(cur_cursor):
                from msit_llm.transform.float_atb_to_quant_atb import transform_quant_cpp_layer_function

                cur_updates = transform_quant_cpp_layer_function.TransformQuantCppLayerFunction(
                    contents, cur_cursor, self.in_tensor_added, indent=self.indent, enable_sparse=self.enable_sparse
                )()
                updates.extend(cur_updates)
                continue
            else:
                continue
            updates.append((insert_start, insert_end, insert_contents))
            print_update_info(insert_contents, insert_start, insert_end)
        return update_contents(contents, updates)


def transform_quant(source_path, enable_sparse=False):
    from glob import glob

    check_libclang_so()
    if source_path.endswith(".cpp"):
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Provided cpp file {source_path} not exists")
        file_list = [source_path]
    else:
        file_list = glob(os.path.join(source_path, "*.cpp"))

    pairs = []
    for cpp_file in file_list:
        if "quant" in os.path.basename(cpp_file):
            continue
        hpp_file = os.path.splitext(cpp_file)[0] + ".h"
        if os.path.exists(hpp_file):
            pairs.append((cpp_file, hpp_file))

    source_files, target_files = [], []
    for cpp_file_path, hpp_file_path in pairs:
        logger.info("cpp_file_path: %r, hpp_file_path: %r" % (cpp_file_path, hpp_file_path))
        target_cpp_file_path, target_hpp_file_path = TransformQuant(
            cpp_file_path, hpp_file_path, enable_sparse=enable_sparse
        )()

        logger.info("\nsource cpp file: %r,\ntarget cpp file: %r" % (cpp_file_path, target_cpp_file_path))
        source_files.append(cpp_file_path)
        target_files.append(target_cpp_file_path)

        logger.info("\nsource hpp file: %r,\ntarget hpp file: %r" % (hpp_file_path, target_hpp_file_path))
        source_files.append(hpp_file_path)
        target_files.append(target_hpp_file_path)

    logger.info(
        "\nTransformed source files: [\n    %s\n]"
        "\nTransformed target files: [\n    %s\n]",
        "\n    ".join("%r" % s for s in source_files),
        "\n    ".join("%r" % t for t in target_files)
    )
