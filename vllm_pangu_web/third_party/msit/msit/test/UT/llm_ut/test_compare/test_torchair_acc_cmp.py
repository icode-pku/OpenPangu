import os
import stat
import shutil
from unittest.mock import patch

import pytest
import numpy as np

import msit_llm
from msit_llm.compare import torchair_acc_cmp


FILE_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
FAKE_GE_DUMP_DATA_NAME = "test_torchair_acc_cmp_fake_ge_dump_data"
FAKE_FX_DUMP_DATA_NAME = "test_torchair_acc_cmp_fake_fx_dump_data"
FAKE_PBTXT_FILE_NAME = torchair_acc_cmp.GE_GRAPH_FILE_PREFIX + "17065969118878_test.txt"
FAKE_PBTXT_FILE_PATH = os.path.join(FAKE_GE_DUMP_DATA_NAME, FAKE_PBTXT_FILE_NAME)


@pytest.fixture(scope="function")
def set_fake_parse_torchair_dump_data():
    def fake_parse_torchair_dump_data(my_path):
        return [np.ones([2, 3]).astype(np.float32)], [np.ones([2, 3]).astype(np.float32)]

    setattr(torchair_acc_cmp, "parse_torchair_dump_data", fake_parse_torchair_dump_data)


@pytest.fixture(scope="function")
def set_fake_set_msaccucmp_path_from_cann():
    setattr(torchair_acc_cmp, "set_msaccucmp_path_from_cann", lambda: None)


@pytest.fixture(scope="function")
def fake_pbtxt_file():
    contents = """
    op {
      name: "Add_2"
      output_desc {
        name: "test"
        attr {
          key: "_fx_tensor_name"
          value {
            s: "mm-aten.mm.default.OUTPUT.0"
          }
        }
        attr {
          name: "tt2"
        }
      }
    }
    op {
      name: "Cast_9"
      output_desc {
        name: "test"
        attr {
          key: "_fx_tensor_name"
          value {
            s: "mm-aten.mm.default.OUTPUT.0"
          }
        }
        attr {
          name: "tt2"
        }
      }
    }"""

    with os.fdopen(os.open(FAKE_PBTXT_FILE_PATH, os.O_CREAT | os.O_WRONLY, FILE_PERMISSION), "w") as ff:
        ff.write(contents)
    yield
    if os.path.exists(FAKE_PBTXT_FILE_PATH):
        os.remove(FAKE_PBTXT_FILE_PATH)


@pytest.fixture(scope="function")
def fake_ge_dump_data():
    base_path = os.path.join(FAKE_GE_DUMP_DATA_NAME, "0")
    os.makedirs(base_path, mode=0o750, exist_ok=True)

    file_names = [
        "Add.Add_2.44.6.17065969121619",
        "Cast.Cast_9.19.6.17065969118878",
        "ConcatV2D.ConcatV2.42.6.17065969121611",
        "TransData.TransData_1.1.42.6.17005969121581",
        torchair_acc_cmp.FUSION_OP_TYPE + ".Add_2Cast_9ConcatV2.19.6.17065969118878",  # Fused op name
    ]
    for file_name in file_names:
        file_path = os.path.join(base_path, file_name)
        with os.fdopen(os.open(file_path, os.O_CREAT | os.O_WRONLY, FILE_PERMISSION), "wb") as ff:
            pass
    yield
    if os.path.exists(FAKE_GE_DUMP_DATA_NAME):
        shutil.rmtree(FAKE_GE_DUMP_DATA_NAME)


@pytest.fixture(scope="function")
def fake_fx_dump_data():
    base_path = os.path.join(FAKE_FX_DUMP_DATA_NAME, "1/foo")
    os.makedirs(base_path, mode=0o750, exist_ok=True)

    file_names = [
        "mm-aten.mm.default.INPUT.0.20240125031118787351.npy",
        "mm-aten.mm.default.INPUT.1.20240125031118787351.npy",
        "mm-aten.mm.default.OUTPUT.0.20240125031118787351.npy",
    ]
    for file_name in file_names:
        np.save(os.path.join(base_path, file_name), np.zeros([]))

    yield

    if os.path.exists(FAKE_FX_DUMP_DATA_NAME):
        shutil.rmtree(FAKE_FX_DUMP_DATA_NAME)


def test_init_ge_dump_data_from_bin_path_given_path_when_valid_then_pass(set_fake_set_msaccucmp_path_from_cann,
                                                                         fake_ge_dump_data,
                                                                         fake_fx_dump_data,
                                                                         fake_pbtxt_file,
                                                                         set_fake_parse_torchair_dump_data
                                                                         ):
    result = torchair_acc_cmp.init_ge_dump_data_from_bin_path(FAKE_GE_DUMP_DATA_NAME)
    fused_op_name = torchair_acc_cmp.FUSION_OP_TYPE + ".Add_2Cast_9ConcatV2.19.6.17065969118878"
    expected_result = [
        {
            0: {
                "Add_2": os.path.join(FAKE_GE_DUMP_DATA_NAME, "0", "Add.Add_2.44.6.17065969121619"),
                "Cast_9": os.path.join(FAKE_GE_DUMP_DATA_NAME, "0", "Cast.Cast_9.19.6.17065969118878"),
                "ConcatV2": os.path.join(FAKE_GE_DUMP_DATA_NAME, "0", "ConcatV2D.ConcatV2.42.6.17065969121611"),
                "TransData_1.1": os.path.join(
                    FAKE_GE_DUMP_DATA_NAME, "0", "TransData.TransData_1.1.42.6.17005969121581"
                ),
                "Add_2Cast_9ConcatV2": os.path.join(FAKE_GE_DUMP_DATA_NAME, "0", fused_op_name),
            }
        }
    ]
    assert result == expected_result


def test_init_fx_dump_data_from_path_given_path_when_valid_then_pass(set_fake_set_msaccucmp_path_from_cann,
                                                                     fake_ge_dump_data,
                                                                     fake_fx_dump_data,
                                                                     fake_pbtxt_file,
                                                                     set_fake_parse_torchair_dump_data
                                                                     ):
    result = torchair_acc_cmp.init_fx_dump_data_from_path(FAKE_FX_DUMP_DATA_NAME)
    expected_result = [
        {
            0: {
                "mm-aten.mm.default": {
                    "input": [
                        os.path.join(
                            FAKE_FX_DUMP_DATA_NAME, "1/foo", "mm-aten.mm.default.INPUT.0.20240125031118787351.npy"
                        ),
                        os.path.join(
                            FAKE_FX_DUMP_DATA_NAME, "1/foo", "mm-aten.mm.default.INPUT.1.20240125031118787351.npy"
                        ),
                    ],
                    "output": [
                        os.path.join(
                            FAKE_FX_DUMP_DATA_NAME, "1/foo", "mm-aten.mm.default.OUTPUT.0.20240125031118787351.npy"
                        )
                    ],
                }
            }
        }
    ]
    assert result == expected_result
