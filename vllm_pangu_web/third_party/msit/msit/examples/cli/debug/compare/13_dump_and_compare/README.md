# dump_data 代码插入使用说明
## 使用方式
使用方式分为两步，**dump数据**和**compare数据**，具体如下：
- 工具对外提供dump_data函数用于数据dump，需要插入在模型脚本中。
- 数据dump结束后会在指定目录生成两个json文件，将对应json的路径作入参输入到工具完成比对。

工具命令如下：

```msit debug compare aclcmp --golden-path path_to_golden_data.json --my-path path_to_acl_data.json --output output_dir```

### dump数据
#### 函数原型
```
dump_data(data_src, data_id, data_val, tensor_path, token_id)
```
#### 功能说明
- 函数实现将需要dump的数据落盘(high-level落盘路径为```当前目录/{PID}_cmp_dump_data/{data_src}_tensor/{token_id}/xxx.npy```)
- 同时将数据信息写到一个```metadata.json```中，用于后续比对（路径为```当前目录/{PID}_cmp_dump_data/{data_src}_tensor/metadata.json```)

#### 参数说明
|参数名         | 说明                                                      |是否必选|
| ------------- |---------------------------------------------------------- | -------- |
|data_src| 用于标记dump数据是标杆数据（CPU/GPU/NPU）还是加速库待比对数据， 取值为```"golden"```或```"acl"```，```"golden"```代表标杆数据，```"acl"```代表加速库待比对数据|是|
|data_id| 数据的**唯一标识**，用于与另一侧数据的data_id匹配（即对应关系），以实现数据比对|是|
|data_val| 针对high-level场景下可直接传入变量名，获得tensor后落盘，high-level场景必选| 否|
|tensor_path| 针对low-level场景下无法直接获取变量名，需要按照加速库dump的目录结构路径进行设置， low-level场景必选| 否|
|token_id| high-level场景下会生成一个带token_id的目录，在该目录下生成dump数据，low-level场景下控制生成指定token_id轮次的数据|是|

**使用注意**
- data_id是数据匹配的唯一标识，需匹配对应。
- tensor_path的目录设置参考形式```{model_index}_{model_name}/{layer_index}_{layer_name}/{op_index}_{op_name}/after/Output0.bin```，如
```"0_ChatGlm6BModelEncoderTorch/0_ChatGlm6BLayerEncoderOperationGraphRunner/3_SelfAttentionOpsChatglm6bRunner/after/outTensor0.bin"```
- token_id只用作数据区分不用作数据标识（low-level场景下只dump指定token轮次的加速库数据），需要用户设置，建议是在单一Encoder/Decoder结构下可直接自增，
Encoder-Decoder结构下区分自增或者针对首词推理针对处理。

#### 使用示例
##### 1.模型代码添加
- 在模型py文件中文件开头导入对应函数
```
from msquickcmp.pta_acl_cmp.compare import dump_data
```

##### 在需要比对的数据位置插入dump_data代码
- 由于dump_data存在high-level和low-level的场景，分别说明：

###### high-level插入代码
```
# 在你需要比对数据的地方
# =============================golden================================
# 分别表示全局自增data_id和token轮次
global data_id, token_id
for i, layer in enumerate(self.layers):

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    layer_ret = layer(
        hidden_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
        layer_id=torch.tensor(i),
        layer_past=past_key_values[i],
        use_cache=use_cache,
        output_attentions=output_attentions
    )

    hidden_states = layer_ret[0]

    if use_cache:
        presents = presents + (layer_ret[1],)

    if output_attentions:
        all_self_attentions = all_self_attentions + \
            (layer_ret[2 if use_cache else 1],)
# 比对最后输出
dump_data("golden", data_id, hidden_states, token_id=token_id)
data_id += 1
# =============================golden================================



# =============================acl================================
# 含义同上
global data_id, token_id
acl_model_out = self.acl_encoder_operation.execute(
    hidden_states, position_ids, self.cos, self.sin, attention_mask, seq_len)
# 比较最后encoder输出
dump_data("acl", data_id, acl_model_out[0], token_id=token_id)
data_id += 1
# =============================acl================================
```


###### low-level插入代码
```
# 在你需要比对数据的地方
# =============================golden================================
# PT侧的low-level只需要在对应函数调用的声明内部加入dump_data即可
def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids,
            attention_mask: torch.Tensor,
            layer_id,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
    # 分别表示全局自增data_id和token轮次
    global data_id, token_id
    """
    Other codes
    """
    # Residual connection.
    alpha = (2 * self.num_layers) ** 0.5
    hidden_states = attention_input * alpha + attention_output

    mlp_input = self.post_attention_layernorm(hidden_states)
    dump_data("golden", data_id, mlp_input, token_id=token_id)
    data_id += 1
# =============================golden================================



# =============================acl================================
# 加速库侧low-level需要在execute调用前先调用dump_data告诉加速库需要dump什么数据
# 分别表示全局自增data_id和token轮次
global data_id, token_id
if past_key_values[0] is None:
    for i in range(num_layers):
        tensor_path = "0_ChatGlm6BModelEncoderTorch/{i}_ChatGlm6BLayerEncoderOperationGraphRunner/6_NormOpsRunner/after/outTensor0.bin"
        dump_data("acl", data_id, tensor_path=tensor_path, token_id=token_id)
        data_id += 1
    acl_model_out = self.acl_encoder_operation.execute(
        hidden_states, position_ids, self.cos, self.sin, attention_mask, seq_len)
# =============================acl================================
```
##### 2.模型推理
###### 加速库运行环境准备
```
source /usr/local/Ascend/ascend-toolkit/set_env
cd output/acltransformer # 加速库目录位置
source set_env.sh

# 需要dump加速库low-level数据时使用
site_packages_path=$(python3 -c "import site; print(site.getsitepackages()[0])")
export LD_PRELOAD="${site_packages_path}/msquickcmp/libsavetensor.so":$LD_PRELOAD
export ATB_SAVE_TENSOR=1 # 打开加速库dump开关，0为关闭dump功能（默认），1为开启；
export ATB_SAVE_TENSOR_START=0 # 加速库接口dump开始数据轮数；
export ATB_SAVE_TENSOR_END=1 # 加速库接口dump结束数据轮数；
export ATB_SAVE_TENSOR_RUNNER=NormOpsRunner # 加速库dump数据runner白名单，默认为空，空时全量dump；非空时只dump白名单包含runner数据。
```

###### 模型推理执行
按照原运行方式运行即可
- 示例：
```
bash run.sh patches/model/modeling_chatglm_model.py
```

##### 3.数据比对
执行完成后数据会落盘，同时生成一个metadata.json（路径为```当前目录/{PID}_cmp_dump_data/{data_src}_tensor/metadata.json```)，将两侧对应的metadata.json路径传入工具入参并指定结果输出路径```output_dir```完成比对。
命令如下：
```msit debug compare aclcmp --golden-path path_to_golden_data.json --my-path path_to_acl_data.json --output output_dir```

完成比对后在```output_dir```下会生成一个```cmp_report.csv```,保存比对的最终结果。
- 比对结果：
![cmp_report.csv](./cmp_report.png)