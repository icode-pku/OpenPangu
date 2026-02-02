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

import torch
import torchvision.models as models
import mindietorch

from msit_llm import DumpConfig, register_hook

# 创建随机输入数据，尺寸为（1, 3, 224, 224）对应于（batch_size, channels, height, width）
input_data = torch.randn(1, 3, 224, 224)

# 加载 ResNet-50 模型
model = models.resnet50(pretrained=True)
model.eval()  # 设置模型为评估模式

dump_config = DumpConfig(dump_path="./dump/cpu")
register_hook(model, dump_config)  # model是要dump中间tensor的模型实例，在模型初始化后添加代码
result_cpu = model.forward(input_data)

# compile 模型
input_info = [mindietorch.Input((1, 3, 224, 224))]
traced_model = torch.jit.trace(model, input_data)
compiled_module = mindietorch.compile(traced_model, inputs=input_info, soc_version="<soc_version>")  # 替换成用户的具体型号

mindietorch.set_device(0)
result = compiled_module(input_data.to("npu:0")).cpu()
mindietorch.finalize()
