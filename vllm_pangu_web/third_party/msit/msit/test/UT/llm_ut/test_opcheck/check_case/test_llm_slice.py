import sys
import pytest
from unittest.mock import MagicMock


# Mocking the OperationTest class to avoid errors
class MockOperationTest:
    def __init__(self, *args, **kwargs):
        self.op_param = kwargs.get('op_param', {})
    
    @staticmethod
    def validate_param(self, *args, **kwargs):
        # For simplicity, assume parameters are valid unless specified otherwise
        return True
    
    def execute(self):
        # Mock execution; do nothing
        pass


@pytest.fixture(scope="function")
def import_allgather_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case.slice import OpcheckSliceOperation
    OpcheckSliceOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckSliceOperation": OpcheckSliceOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_test_given_invalid_params_when_validate_param_fails_then_not_execute(import_allgather_module):
    OpcheckSliceOperation = import_allgather_module['OpcheckSliceOperation']
    op = OpcheckSliceOperation()
    op.validate_param = MagicMock(return_value=False)
    op.execute = MagicMock()

    op.test()

    op.execute.assert_not_called()


def test_test_method_given_valid_params_when_called_then_execute_called(import_allgather_module):
    OpcheckSliceOperation = import_allgather_module['OpcheckSliceOperation']
    op = OpcheckSliceOperation(op_param={'offsets': [1, 2], 'size': [3, 4]})
    op.execute = MagicMock()

    op.test()
    
    op.execute.assert_called_once()


def test_test_method_given_invalid_params_when_called_then_execute_not_called(import_allgather_module):
    OpcheckSliceOperation = import_allgather_module['OpcheckSliceOperation']
    op = OpcheckSliceOperation(op_param={'offsets': [1], 'size': [2]})
    op.validate_param = MagicMock(return_value=False)
    op.execute = MagicMock()

    op.test()

    op.execute.assert_not_called()
