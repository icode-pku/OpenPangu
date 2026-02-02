from unittest.mock import patch, MagicMock
import pytest
import torch
from msit_llm.transform.torch_to_atb_python.run import CausalLM, Runner, MODEL_PATH

# Mocking required modules and functions
@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch('msit_llm.transform.torch_to_atb_python.run.load_model_dict'), \
         patch('msit_llm.transform.torch_to_atb_python.run.ATBModel'), \
         patch('msit_llm.transform.torch_to_atb_python.run.AutoConfig'), \
         patch('msit_llm.transform.torch_to_atb_python.run.Model'), \
         patch('msit_llm.transform.torch_to_atb_python.run.PreTrainedModel.__init__'), \
         patch('msit_llm.transform.torch_to_atb_python.run.AutoTokenizer'), \
         patch('msit_llm.transform.torch_to_atb_python.run.CausalLMOutputWithPast'):
        yield

# Test CausalLM initialization
def test_CausalLM_init_given_model_path_when_initialized_then_sets_up_correctly():
    mock_config = MagicMock()
    mock_AutoConfig = MagicMock()
    mock_AutoConfig.from_pretrained.return_value = mock_config

    mock_atb_model = MagicMock()
    mock_ATBModel = MagicMock()
    mock_ATBModel.return_value = mock_atb_model

    mock_weights = MagicMock()
    mock_load_model_dict = MagicMock()
    mock_load_model_dict.return_value = mock_weights

    mock_model = MagicMock()
    mock_Model = MagicMock()
    mock_Model.return_value = mock_model

    with patch('msit_llm.transform.torch_to_atb_python.run.AutoConfig', mock_AutoConfig), \
         patch('msit_llm.transform.torch_to_atb_python.run.ATBModel', mock_ATBModel), \
         patch('msit_llm.transform.torch_to_atb_python.run.load_model_dict', mock_load_model_dict), \
         patch('msit_llm.transform.torch_to_atb_python.run.Model', mock_Model):
        model = CausalLM(MODEL_PATH)

    mock_AutoConfig.from_pretrained.assert_called_once_with(MODEL_PATH)
    mock_ATBModel.assert_called_once_with(mock_model)
    mock_load_model_dict.assert_called_once_with(MODEL_PATH)
    mock_atb_model.set_weights.assert_called_once_with(mock_weights)

# Test CausalLM init_kv_cache method
def test_CausalLM_init_kv_cache_given_initialized_model_when_called_then_initializes_kv_cache():
    mock_atb_model = MagicMock()
    mock_ATBModel = MagicMock()
    mock_ATBModel.return_value = mock_atb_model

    with patch('msit_llm.transform.torch_to_atb_python.run.ATBModel', mock_ATBModel):
        model = CausalLM(MODEL_PATH)
        model.init_kv_cache()

    mock_atb_model.init_kv_cache.assert_called_once()

# Test CausalLM prepare_inputs_for_generation method
def test_CausalLM_prepare_inputs_for_generation_given_input_ids_when_called_then_returns_correct_inputs():
    model = CausalLM(MODEL_PATH)
    input_ids = torch.tensor([[1, 2, 3]])
    past_key_values = None
    attention_mask = torch.tensor([[1, 1, 1]])
    inputs_embeds = None

    inputs = model.prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, inputs_embeds)

    assert 'input_ids' in inputs
    assert 'position_ids' in inputs
    assert 'past_key_values' in inputs
    assert 'use_cache' in inputs
    assert 'attention_mask' in inputs

def test_CausalLM_prepare_inputs_for_generation_given_past_key_values_when_called_then_returns_correct_inputs():
    model = CausalLM(MODEL_PATH)
    input_ids = torch.tensor([[1, 2, 3]])
    past_key_values = (torch.tensor([[1]]), torch.tensor([[1]]))
    attention_mask = torch.tensor([[1, 1, 1]])
    inputs_embeds = None

    inputs = model.prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, inputs_embeds)

    assert 'input_ids' in inputs
    assert 'position_ids' in inputs
    assert 'past_key_values' in inputs
    assert 'use_cache' in inputs
    assert 'attention_mask' in inputs

# Test CausalLM forward method
def test_CausalLM_forward_given_input_ids_when_called_then_returns_correct_output():
    mock_atb_model = MagicMock()
    mock_ATBModel = MagicMock()
    mock_ATBModel.return_value = mock_atb_model
    mock_atb_model.forward.return_value = {"output": torch.tensor([[1.0, 2.0, 3.0]])}

    with patch('msit_llm.transform.torch_to_atb_python.run.ATBModel', mock_ATBModel):
        model = CausalLM(MODEL_PATH)
        input_ids = torch.tensor([[1, 2, 3]])
        position_ids = torch.tensor([[0, 1, 2]])
        use_cache = False

        output = model.forward(input_ids, position_ids, use_cache)

    assert isinstance(output, MagicMock)

def test_CausalLM_forward_given_use_cache_when_called_then_returns_correct_output():
    mock_atb_model = MagicMock()
    mock_ATBModel = MagicMock()
    mock_ATBModel.return_value = mock_atb_model
    mock_atb_model.forward.return_value = {"output": torch.tensor([[1.0, 2.0, 3.0]])}

    with patch('msit_llm.transform.torch_to_atb_python.run.ATBModel', mock_ATBModel):
        model = CausalLM(MODEL_PATH)
        input_ids = torch.tensor([[1, 2, 3]])
        position_ids = torch.tensor([[0, 1, 2]])
        use_cache = True

        output = model.forward(input_ids, position_ids, use_cache)

    assert isinstance(output, MagicMock)

# Test Runner initialization
def test_Runner_init_given_model_path_when_initialized_then_sets_up_correctly():
    mock_tokenizer = MagicMock()
    mock_AutoTokenizer = MagicMock()
    mock_AutoTokenizer.from_pretrained.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_CausalLM = MagicMock()
    mock_CausalLM.return_value = mock_model

    with patch('msit_llm.transform.torch_to_atb_python.run.AutoTokenizer', mock_AutoTokenizer), \
         patch('msit_llm.transform.torch_to_atb_python.run.CausalLM', mock_CausalLM):
        runner = Runner(MODEL_PATH)

    mock_AutoTokenizer.from_pretrained.assert_called_once_with(MODEL_PATH)
    mock_CausalLM.assert_called_once_with(MODEL_PATH)
    assert runner.max_input_length == 20
    assert runner.max_output_length == 20
    assert runner.batch_size == 1

# Test Runner warm_up method
def test_Runner_warm_up_given_initialized_runner_when_called_then_warms_up_model():
    mock_model = MagicMock()
    mock_CausalLM = MagicMock()
    mock_CausalLM.return_value = mock_model

    with patch('msit_llm.transform.torch_to_atb_python.run.CausalLM', mock_CausalLM):
        runner = Runner(MODEL_PATH)
        runner.warm_up()

    mock_model.generate.assert_called_once()

# Test Runner infer method
def test_Runner_infer_given_input_text_when_called_then_returns_correct_output():
    mock_model = MagicMock()
    mock_CausalLM = MagicMock()
    mock_CausalLM.return_value = mock_model

    mock_tokenizer = MagicMock()
    mock_AutoTokenizer = MagicMock()
    mock_AutoTokenizer.from_pretrained.return_value = mock_tokenizer

    mock_tokenizer.return_value.input_ids.npu.return_value = 'input_ids'
    mock_tokenizer.return_value.attention_mask.npu.return_value = 'attention_mask'

    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
    mock_tokenizer.batch_decode.return_value = ["output_text"]

    with patch('msit_llm.transform.torch_to_atb_python.run.CausalLM', mock_CausalLM), \
         patch('msit_llm.transform.torch_to_atb_python.run.AutoTokenizer', mock_AutoTokenizer):
        runner = Runner(MODEL_PATH)
        output_text = runner.infer("Who's there?")

    assert output_text == "output_text"

def test_Runner_infer_given_input_text_when_called_with_use_cache_false_then_returns_correct_output():
    mock_model = MagicMock()
    mock_CausalLM = MagicMock()
    mock_CausalLM.return_value = mock_model

    mock_tokenizer = MagicMock()
    mock_AutoTokenizer = MagicMock()
    mock_AutoTokenizer.from_pretrained.return_value = mock_tokenizer

    mock_tokenizer.return_value.input_ids.npu.return_value = 'input_ids'
    mock_tokenizer.return_value.attention_mask.npu.return_value = 'attention_mask'

    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
    mock_tokenizer.batch_decode.return_value = ["output_text"]

    with patch('msit_llm.transform.torch_to_atb_python.run.CausalLM', mock_CausalLM), \
         patch('msit_llm.transform.torch_to_atb_python.run.AutoTokenizer', mock_AutoTokenizer):
        runner = Runner(MODEL_PATH)
        output_text = runner.infer("Who's there?", use_cache=False)

    assert output_text == "output_text"