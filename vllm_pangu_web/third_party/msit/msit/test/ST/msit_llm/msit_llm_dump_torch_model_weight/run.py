# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import os.path

from main_torch import check_min_version, override_topp_and_topk, parse_args, get_model, precision, performance, \
    cli_demo, webUI


def main():
    # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
    check_min_version('4.30.2')
    override_topp_and_topk()
    args = parse_args()
    tokenizer, model = get_model(args)
    from msit_llm import DumpConfig, register_hook

    dump_config = DumpConfig(dump_path=os.path.realpath("./ait_dump_torch_weight"),
                            dump_weight=True, mode=["module", "api"],
                            layer_name="root.transformer.encoder.layers.0*")
    register_hook(model, dump_config)

    if 'precision' in args.mode:
        precision(args, tokenizer, model)
    elif 'performance' in args.mode:
        performance(args, tokenizer, model)
    elif 'cli_demo' in args.mode:
        cli_demo(args, tokenizer, model)
    else:
        webUI(args, tokenizer, model)


if __name__ == '__main__':
    main()