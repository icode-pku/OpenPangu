import os
import shutil
import sys
import pytest
from unittest.mock import patch, MagicMock

from msit_llm.dump.torchair_dump.torchair_dump import try_import_torchair, get_ge_dump_config, get_fx_dump_config


def test_try_import_torchair_given_missing_torch_when_called_then_error():
    with patch.dict(sys.modules, {'torch': None}):
        with pytest.raises(ModuleNotFoundError):
            try_import_torchair()


@patch('time.strftime', return_value='20250414_120000')
@patch('components.utils.security_check.ms_makedirs')
@patch('msit_llm.common.utils.check_output_path_legality')
def test_get_ge_dump_config_given_valid_params_when_called_then_success(
    mock_check, mock_mkdir, mock_time
):
    mock_torchair = MagicMock()
    mock_torchair.configs.compiler_config = MagicMock()
    mock_torch_npu = MagicMock()
    with patch.dict('sys.modules', {
             'torchair': mock_torchair,
             'torchair.configs.compiler_config': mock_torchair.configs.compiler_config,
             'torch_npu': mock_torch_npu,
         }):
        dump_path = os.path.dirname(os.path.realpath(__file__))

        config = get_ge_dump_config(
            dump_path=dump_path,
            fusion_switch_file=os.path.realpath(__file__),
            dump_token=[1, 2],
            dump_layer=["conv"]
        )
        shutil.rmtree(os.path.join(dump_path, 'msit_ge_dump'))
        assert config.dump_config.enable_dump
        assert config.fusion_config.fusion_switch_file == os.path.realpath(__file__)
        assert config.dump_config.dump_step == "1|2"
        assert config.dump_config.dump_layer == "conv"


def test_get_fx_dump_config_given_default_when_called_then_npy_type():
    mock_torchair = MagicMock()
    mock_torchair.configs.compiler_config = MagicMock()
    mock_torch_npu = MagicMock()
    with patch.dict('sys.modules', {
             'torchair': mock_torchair,
             'torchair.configs.compiler_config': mock_torchair.configs.compiler_config,
             'torch_npu': mock_torch_npu,
         }):
        config = get_fx_dump_config()
        assert config.debug.data_dump.type == "npy"
