
### 1. GE开启融合（默认） Dump 案例
  ```py
  import os, numpy as np
  import torch, torch_npu, torchair as tng, torchvision
  from msit_llm.dump import torchair_dump

  target_dtype = torch.float16
  model = torchvision.models.resnet50(pretrained=True).eval().to(target_dtype).npu()
  if not os.path.exists('aa_224_224.npy'):
      np.save('aa_224_224.npy', np.random.uniform(size=[1,3,224,224]))
  aa = torch.from_numpy(np.load('aa_224_224.npy')).to(target_dtype).npu()
  config = torchair_dump.get_ge_dump_config(dump_path="dump")
  npu_backend = tng.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  with torch.no_grad():
      try:
          print(model(aa).shape)
      except:
          pass
  ```

### 2. FX Dump 案例
  ```py
  import os, numpy as np
  import torch, torch_npu, torchair as tng, torchvision
  from msit_llm.dump import torchair_dump

  target_dtype = torch.float16
  model = torchvision.models.resnet50(pretrained=True).eval().to(target_dtype).npu()
  if not os.path.exists('aa_224_224.npy'):
      np.save('aa_224_224.npy', np.random.uniform(size=[1,3,224,224]))
  aa = torch.from_numpy(np.load('aa_224_224.npy')).to(target_dtype).npu()
  config = torchair_dump.get_fx_dump_config()
  npu_backend = tng.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  with torch.no_grad():
      try:
          print(model(aa).shape)
      except:
          pass
  ```

### 3. GE关闭融合 Dump 案例
  ```py
  import os, numpy as np
  import torch, torch_npu, torchair as tng, torchvision
  from msit_llm.dump import torchair_dump

  target_dtype = torch.float16
  model = torchvision.models.resnet50(pretrained=True).eval().to(target_dtype).npu()
  if not os.path.exists('aa_224_224.npy'):
      np.save('aa_224_224.npy', np.random.uniform(size=[1,3,224,224]))
  aa = torch.from_numpy(np.load('aa_224_224.npy')).to(target_dtype).npu()
  config = torchair_dump.get_ge_dump_config(dump_path="dump", fusion_switch_file='./fusion_switch.json')
  npu_backend = tng.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  with torch.no_grad():
      try:
          print(model(aa).shape)
      except:
          pass
  ```
  新建fusion_switch.json文件，内容如下：
  ```json
  {
    "Switch": {
      "GraphFusion": {
        "ALL": "off"
      },
      "UBFusion": {
        "ALL": "off"
      }
    }
  }
  ```