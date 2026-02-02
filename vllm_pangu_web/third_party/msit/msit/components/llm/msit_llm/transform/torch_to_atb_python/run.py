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

import torch
import torch_npu
from transformers import AutoTokenizer, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from msit_llm.transform.torch_to_atb_python import ATBModel
from msit_llm.transform.utils import load_model_dict, write_file
from msit_llm.common.utils import check_input_path_legality

MODEL_PATH = "model_path_placeholder"


class CausalLM(PreTrainedModel):
    def __init__(self, model_path):
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        super().__init__(config)

        device = torch.device(f"npu")
        torch.npu.set_device(device)
        self.placeholder = torch.zeros(1, device=device)

        self.atb_model = ATBModel(Model())
        weights = load_model_dict(model_path)
        self.atb_model.set_weights(weights)

    def init_kv_cache(self):
        self.atb_model.init_kv_cache()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(self, input_ids, position_ids, use_cache=False, **kwargs):
        if input_ids.dim() == 2:
            input_ids = input_ids[0]

        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[0]).to(input_ids)
   
        if position_ids.dim() == 2:
            position_ids = position_ids[0]

        if use_cache:
            past_key_values = (self.placeholder, self.placeholder)
        else:
            past_key_values = None
            self.atb_model.init_kv_cache()

        if "position_ids" in self.atb_model.inputs:
            out = self.atb_model.forward(input_ids=input_ids, position_ids=position_ids)
        else:
            out = self.atb_model.forward(input_ids=input_ids)

        logits = out["output"].unsqueeze(0)

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


class Runner:
    def __init__(self, model_path):
        self.model = CausalLM(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.max_input_length = 20
        self.max_output_length = 20
        self.batch_size = 1

    def warm_up(self):
        dummy_input_ids_full = torch.randint(0, 32000, [self.batch_size, self.max_input_length], dtype=torch.long).npu()
        self.model.generate(inputs=dummy_input_ids_full, do_sample=False, max_new_tokens=10)

    def infer(self, input_text, use_cache=True):
        if isinstance(input_text, str):
            input_text = [input_text] * self.batch_size

        inputs = self.tokenizer(input_text, return_tensors="pt")

        self.model.init_kv_cache()

        # Prefill
        with torch.no_grad():
            generate_ids = self.model.generate(
                inputs=inputs.input_ids.npu(), attention_mask=inputs.attention_mask.npu(), max_new_tokens=1
            )

        # Deocde
        with torch.no_grad():
            generate_ids = self.model.generate(
                inputs=inputs.input_ids.npu(),
                attention_mask=inputs.attention_mask.npu(),
                max_new_tokens=self.max_output_length,
                use_cache=use_cache,
            )

        generate_text = self.tokenizer.batch_decode(
            generate_ids[:, :], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return generate_text[0][:]


if __name__ == "__main__":
    import sys
    import logging
    import argparse
    mindie_supported = False
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-w", "--weight", type=check_input_path_legality, default=MODEL_PATH, help="Model weight path")
    parser.add_argument("-i", "--inputs", type=str, default="Who's there?", help="input for model")
    args = parser.parse_known_args()[0]
    run_pa_path = "xxx"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.info("Build it yourself Please ensure that the input is legal and compliant")
    if mindie_supported:
        import subprocess
        from pathlib import Path
        run_pa_path = check_input_path_legality(run_pa_path)
        contents_str = Path(run_pa_path).read_text()
        contents_str = contents_str.replace("Who is the CEO of Google?", args.inputs)
        write_file(run_pa_path, contents_str)
        command = ["bash", run_pa_path, args.weight]
        result = subprocess.run(command, shell=False)
        contents_str = Path(run_pa_path).read_text()
        contents_str = contents_str.replace(args.inputs, "Who is the CEO of Google?")
        write_file(run_pa_path, contents_str)
    else:
        from atb_model_placeholder import Model

        runner = Runner(args.weight)
        runner.warm_up()
        output_text = runner.infer(args.inputs, use_cache=False)

        
        logger.info("-" * 40)
        logger.info("Input: %r", args.inputs)
        logger.info("Output: %s", output_text)
