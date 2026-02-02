

# set_label代码插入使用说明

大模型加速库精度比对是以PyTorch Ascend(pta)侧的数据作为基准数据，比对加速库(acl)推理的数据与pta数据之间的差异，辅助开发者找出加速库侧的问题Operation。
## 1. 比对level
加速库的Operation分为3个粒度：Op、Layer、Model，在加速库开发过程中pta侧代码的替换也会分为这3个粒度。
## 1.1 Op的替换
若是Op粒度的替换，pta侧可以获取Operation的输入/输出数据，一个Operation内部会有多个kernel，pta侧无法获取到Op内部kernel的数据，因此需要加速库侧提供数据。<br>

如果比较Operation的输入/输出数据的精度，则称为***high-level***。如果比较Operation内部kernel的输入/输出数据的精度，则称为***low-level***。

## 1.2 Layer的替换

若是layer粒度的替换，pta侧可以获取整个Layer的输入/输出数据，一个Layer内部会有多个operation或kernel，pta侧无法获取到Layer内部Operation或者kernel的数据，因此需要加速库侧提供数据。

如果比较Layer的输入/输出数据的精度，则称为***high-level***。如果比较Layer内部kernel的输入/输出数据的精度，则称为***low-level***。

## 1.3 Model的替换

若是model粒度的替换，pta侧可以获取整个Model的输入/输出数据，一个Model内部会有多个Layer或Operation，pta侧无法获取Model内部Layer或者Operation的数据，因此需要加速库侧提供数据。

如果比较Model的输入/输出数据的精度，则称为***high-level***。如果比较Model内部Layer或者Operation的输入/输出数据的精度，则称为***low-level***。

## 2. 接口介绍

## 2.1 API介绍

### 2.1 set_label

set_label(data_src, data_id, data_val, tensor_path)

接口描述：用于在模型pta侧代码中打标签，记录待比对数据的来源、id、值以及路径。

返回值：无。

| 参数名      | 含义                   | 是否必填 | 使用说明                                                     |
| ----------- | ---------------------- | -------- | ------------------------------------------------------------ |
| data_src    | 数据来源               | 是       | 数据类型：str。可选值：acl、pta。acl表示加速库的数据，pta表示PyTorch Ascend的数据，即基准数据。 |
| data_id     | 数据的id。             | 是       | 数据类型：str，通过接口gen_id()生成，id一致的数据表示成对比较的数据。 |
| data_val    | 数据的值。             | 否。     | 数据类型: torch.Tensor。当data_src是pta时，这个值是必填的。当data_src是acl时，若是high-level比对，需要必填。若是low-level比对，不需要填。 |
| tensor_path | 加速库侧数据dump的路径 | 否       | 数据类型：str。当data_src是acl时，进行low-level比对时，需要提供加速库侧dump的operation或kernel的数据路径。 |

### 2.2 gen_id

gen_id()

接口描述：根据时间戳生成数据的id。

返回值：data_id，str类型。

### 2.3 set_task_id

set_task_id()

接口描述：用于设置加速库侧dump的数据目录，建议在每轮对话开始前调用下，可以进行多轮对话的精度比对。

返回值：无。

## 2.2. 命令行介绍

使用格式：

```shell
msit debug compare aclcmp xx_args
```

可选参数如下：

| 参数名 | 含义                                                         |
| ------ | ------------------------------------------------------------ |
| --exec | 执行命令，用于拉起大模型推理脚本。建议使用bash xx.sh args或者python3 xx.py的方式拉起。 |

# 3. 使用示例
使用前请安装msit工具，安装指导参考：https://gitcode.com/Ascend/msit/blob/master/msit/docs/install/README.md 以 chatglm-6b为例，介绍下如何使用加速库精度比对工具。

1.  设置task_id

   在每轮对话开始前设置task_id，修改main_performance.py

   ```
      from msquickcmp.pta_acl_cmp.compare import set_task_id
       while True:
           set_task_id()
           query = input("\n用户：")
           if query.strip() == "stop":
               break
           if query.strip() == "clear":
               history = []
   ```

2. 在pta代码中打标签

   以patches/models/modeling_chatglm_model.py为例，该脚本是model粒度的替换。

   high-level比对，比对model的输出与pta对应的model的输出的精度，找到相应的代码段，添加以下代码

   ```
   from msquickcmp.pta_acl_cmp.compare import set_label, gen_id
   data_id = gen_id()
   set_label("pta", data_id, hidden_states)
   set_label("acl", data_id, acl_model_out[0])
   ```

   low-level比对，比对model内部的SelfAttentionOpsChatglm6bRunner的数据与pta侧相应算子的差异，找到相应的代码段，添加以下代码：

   ```python
   from msquickcmp.pta_acl_cmp.compare import set_label, gen_id
   data_id = gen_id()
   set_label("pta", data_id, data_val=attention_output)
   set_label("acl", data_id, tensor_path="0_ChatGlm6BModelEncoderTorch/"
   "0_ChatGlm6BLayerEncoderOperationGraphRunner"
                   "/3_SelfAttentionOpsChatglm6bRunner/after/outTensor0.bin")
   ```

3. 执行比对命令
   安装加速库的指导文档下载编译好加速库代码，进入example/chatglm6b目录，执行比对命令：

   ```shell
   msit debug compare aclcmp --exec "bash run_performance.sh patches/models/modeling_chatglm_model.py"
   ```
- **注意事项**：
- [ ] 在进行low-level比对时，需要先执行PTA model的推理，再执行加速库的推理，即`set_label("acl", data_id, tensor_path)`需要在加速库侧推理前执行。
- [ ] low-level比对的同时，需要设置high-level比对。

4. 结果分析

   生成的csv报告如下：
    ![输入图片说明](%E6%AF%94%E5%AF%B9%E6%8A%A5%E5%91%8A.PNG)

   csv各列名称解释如下：

   | 列名                        | 含义                       |
   | --------------------------- | -------------------------- |
   | data_id                     | 数据的id                   |
   | pta_data_path               | pta数据的dump路径          |
   | pta_dtype                   | pta数据的类型              |
   | pta_shape                   | pta数据的shape             |
   | pta_max_value               | pta数据的最大值            |
   | pta_min_value               | pta数据的最小值            |
   | pta_mean_value              | pta数据的平均值            |
   | acl_data_path               | acl数据的dump路径          |
   | acl_dtype                   | acl数据的类型              |
   | acl_shape                   | acl数据的shape             |
   | acl_max_value               | acl数据的最大值            |
   | acl_min_value               | acl数据的最小值            |
   | acl_mean_value              | acl数据的平均值            |
   | cmp_flag                    | 是否比较                   |
   | cosine_similarity           | pta与acl数据的余弦相似度值 |
   | max_relative_error          | pta与acl数据的最大相对误差 |
   | mean_relative_error         | pta与acl数据的绝对相对误差 |
   | relative_euclidean_distance | pta与acl数据的相对欧式距离 |
   | cmp_fail_reason             | 比对失败的原因             |
   
   比对算法解释如下：
   | 比对算法名称                | 说明                                                         |
   | :-------------------------- | :----------------------------------------------------------- |
   | cosine_similarity           | 进行余弦相似度算法比对出来的结果。取值范围为[-1,1]，比对的结果如果越接近1，表示两者的值越相近，越接近-1意味着两者的值越相反。 |
   | max_relative_error          | 表示最大相对误差。取值范围为[0, +∞)，值越接近于0，表明越相近，值越大，表明差距越大。 |
   | mean_relative_error         | 表示平均相对误差。取值范围为[0, +∞)，值越接近于0，表明越相近，值越大，表明差距越大。 |
   | relative_euclidean_distance | 进行欧氏相对距离算法比对出来的结果。取值范围为[0, +∞)，值越接近于0，表明越相近，值越大，表明差距越大。 |
