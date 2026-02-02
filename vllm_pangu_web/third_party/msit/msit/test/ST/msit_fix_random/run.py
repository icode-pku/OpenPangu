# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

import argparse
import logging
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from msit_llm import seed_all


def get_model_output(seed, config):
    """Helper function to get model output for a given seed and config."""
    seed_all(seed=seed)
    model = LlamaForCausalLM(config).eval()
    output = model(torch.arange(32)[None].long())
    return output.logits


def main():
    config = LlamaConfig()
    config.num_hidden_layers = 2
    config.hidden_size = 1024
    config.intermediate_size = 4096

    # Test same seed
    output1 = get_model_output(seed=1, config=config)
    output2 = get_model_output(seed=1, config=config)

    if not torch.equal(output1, output2):
        logging.error("FAIL: Output should be the same for the same seed")
        exit(1)

    # Test different seeds
    output1 = get_model_output(seed=1, config=config)
    output2 = get_model_output(seed=2, config=config)

    if torch.equal(output1, output2):
        logging.error("FAIL: Output should be different for different seeds")
        exit(1)
    
    logging.info("PASS: All tests passed")
    exit(0)


if __name__ == "__main__":
    main()
