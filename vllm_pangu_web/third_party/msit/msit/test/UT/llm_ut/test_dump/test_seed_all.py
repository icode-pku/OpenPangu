import argparse
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from msit_llm import seed_all


def get_moder_output(seed, config):
    """Helper function to get model output for a given seed and config."""
    seed_all(seed=seed)
    model = LlamaForCausalLM(config).eval()
    output = model(torch.arange(32)[None].long())
    # Extract the tensor from the output object
    return output.logits


def test_same_seed_same_output():
    """Test that model produces the same output for the same seed."""
    config = LlamaConfig()
    config.num_hidden_layers = 2
    config.hidden_size = 256
    config.intermediate_size = 1024  # smaller

    output1 = get_moder_output(seed=1, config=config)
    output2 = get_moder_output(seed=1, config=config)

    assert torch.equal(output1, output2), "Output shoule be the same for the same seed"


def test_different_seed_different_output():
    """Test that model produces the different output for the different seeds."""
    config = LlamaConfig()
    config.num_hidden_layers = 2
    config.hidden_size = 256
    config.intermediate_size = 1024  # smaller

    output1 = get_moder_output(seed=1, config=config)
    output2 = get_moder_output(seed=2, config=config)

    assert not torch.equal(output1, output2), "Output shoule be the different for the different seeds"


def test_invalid_seed():
    """Test that an invalid seed raises an appropriate error."""
    config = LlamaConfig()
    config.num_hidden_layers = 2
    config.hidden_size = 256
    config.intermediate_size = 1024  # smaller

    # Test non-integer seed
    with pytest.raises(argparse.ArgumentTypeError):
        seed_all(seed="invalid_seed")
