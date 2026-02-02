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

import logging

import torch
import torch_npu
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoConfig, PreTrainedModel
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector, LlavaForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from msit_llm.transform.torch_to_atb_python import ATBModel
from msit_llm.transform.utils import load_model_dict
from msit_llm.common.utils import check_input_path_legality
from components.utils.utils import load_file_to_read_common_check_for_cli
from msit_llm.common.log import logger
from atb_model_placeholder import Model

TEXT_CONFIG_ATTR_CANDIDATES = ["text_config", "llm_config"]

MODEL_PATH = "model_path_placeholder"

 
class CausalLM(PreTrainedModel):
    def __init__(self, model_path):
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        for attr in TEXT_CONFIG_ATTR_CANDIDATES:
            if hasattr(config, attr):
                config = getattr(config, attr)
                break
        
        super().__init__(config)
        device = torch.device("npu")
        torch.npu.set_device(device)
        self.placeholder = torch.zeros([1, 1, 1, 1], device=device)

        self.atb_model = ATBModel(Model())
        state_dict = load_model_dict(model_path)

        embed_name, self.embed_tokens = None, None
        for weight_name in state_dict.keys():
            split_weight_name = weight_name.split(".")
            if len(split_weight_name) < 3:
                continue
            layer_name = split_weight_name[2]
            if "embed" in layer_name:
                embed_name = weight_name
                self.embed_tokens = state_dict[embed_name]
                break

        if embed_name is not None:
            prefix = embed_name.split(".")[0] + "."
            logging.info("embed_name: %s, prefix: %s", embed_name, prefix)
            state_dict = {kk.replace(prefix, ""): vv for kk, vv in state_dict.items() if kk.startswith(prefix)}
        else:
            logging.warning("embed layer not found in model, may throw error in later using `get_input_embeddings`")
        self.atb_model.set_weights(state_dict)

    def get_input_embeddings(self):
        return lambda input_ids: self.embed_tokens[input_ids]
    
    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        
        use_cache = kwargs.get("use_cache", None)
        past_key_values = kwargs.get("past_key_values", None)

        seq_len = 1
        if inputs_embeds is not None and inputs_embeds.dim() == 3:
            inputs_embeds = inputs_embeds[0]
            seq_len = inputs_embeds.shape[0] 

        if position_ids is None:
            position_ids = torch.arange(seq_len)
        if position_ids is not None and position_ids.dim() == 2:
            position_ids = position_ids[0]

        if use_cache:
            past_key_values = [(self.placeholder, self.placeholder)]
        else:
            past_key_values = None
            self.atb_model.init_kv_cache()

        out = self.atb_model.forward(
            inputs_embeds=inputs_embeds.half().npu(),
            position_ids=position_ids.npu(),
        )

        logits = out["output"].unsqueeze(0).float().cpu()

        return CausalLMOutputWithPast(logits=logits, past_key_values=None)
    

class VLForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = CausalLM(MODEL_PATH)
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    

if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--image", type=load_file_to_read_common_check_for_cli, required=True, help="image path")
    parser.add_argument(
        "-w", "--weight", type=check_input_path_legality, default=MODEL_PATH, help="model weight path"
    )
    parser.add_argument("-t", "--text", type=str, default="Describe the image.", help="input text for model")
    args = parser.parse_known_args()[0]

    MODEL_PATH = args.weight
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = VLForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)

    try:
        with Image.open(args.image) as image:
            text = "USER: <image>\n" + args.text + "ASSISTANT:"
            inputs = processor(text=text, images=image, return_tensors="pt")
    except Exception as e:
        logger.error("An errer occurred: %s", e)
        raise
    
    generate_ids = model.generate(**inputs, max_new_tokens=15)
    output = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    logger.info("-" * 40)
    logger.info("Input: %r" % args.text)
    logger.info("Output: %s", output)