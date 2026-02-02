# openPangu盘古1b/7b模型W8A8量化指南

## 一、概述

确保在你的容器挂载的/**opt/pangu**目录或其他自定义的路径下已经预下载盘古模型权重：
- 盘古1B：/opt/pangu/openPangu-Embedded-1B-V1.1
- 盘古7B：/opt/pangu/openPangu-Embedded-7B-V1.1

## 二、环境配置

确保已经基于[vllm_pangu_web/README.md](../README.md)安装好量化环境配置

## 三、运行量化脚本

运行**vllm_pangu_web/quantization/w8a8_int8_dynamic.py**量化脚本

量化脚本的自定义参数说明
MODEL_ID：原始的盘古模型路径
SAVE_DIR：量化完成后保存的量化模型路径
```
python3 w8a8_int8_dynamic.py
```

## 四、量化模型测试：

运行**vllm_pangu_web/quantization/vllm-inference**目录下的bash脚本即可

示例：

运行盘古7b量化模型
```
cd vllm_pangu_web/quantization/vllm-inference/
bash run_service_7b.sh  # 拉起vllm服务 加载模型
```
拉起完成后，发起对话请求：
```
bash test_7b.sh # 发起对话请求 测试
```

结果示例：
```
{"id":"chatcmpl-4533ae213c874fc5a24494a4418a0584","object":"chat.completion","created":1766027803,"model":"pangu_embedded_7b","choices":[{"index":0,"message":{"role":"assistant","content":" [unused16] Okay, the user asked how to keep healthy and wants the response in Chinese. Let me start by breaking down the main areas of health. Physical health is probably the most important, so I should cover diet, exercise, sleep, and avoiding harmful habits like smoking or drinking too much.\n\nThen there's mental health. Stress management and staying positive are key here. Maybe mention mindfulness or hobbies. Social connections are also important for mental well-being.\n\nPreventive care is another aspect. Regular check-ups, vaccinations, and screenings can help catch issues early. Also, staying hydrated and avoiding unnecessary risks like pollution or accidents.\n\nI should structure the answer in a clear, easy-to-follow way. Use bullet points or sections in Chinese. Make sure the language is simple and practical. Let me check if I missed anything. Oh, maybe include things like regular exercise routines, balanced meals, and adequate rest. Also, maybe touch on avoiding processed foods and excessive sugar.\n\nWait, the user wants the response in Chinese, so I need to translate all that into natural, conversational Chinese. Avoid technical terms where possible. Keep it friendly and encouraging. Let me make sure each section is concise but covers the essentials. Maybe start with an introduction, then list the main points with explanations. End with a positive note to motivate the user. [unused17] 保持健康的关键在于平衡的生活方式，以下是一些实用建议：\n\n### 1. **均衡饮食**\n   - **多样化摄入**：每天摄入蔬菜、水果、全谷物、优质蛋白（如鱼、豆类、瘦肉）和健康脂肪（如坚果、橄榄油）。\n   - **少加工少添加**：减少高糖、高盐、高脂肪的加工食品，避免含糖饮料和零食。\n   - **规律三餐**：避免暴饮暴食，可适当加健康小食（如酸奶、水果）。\n\n### 2. **坚持运动**\n   - **每周150分钟**：如快走、游泳、骑自行车等有氧运动，结合力量训练（如举重、深蹲）。\n   - **日常活动**：多走路、爬楼梯，减少久坐时间。\n\n### 3. **充足睡眠**\n   - **7-9小时/天**：保持规律作息，睡前避免电子设备，营造安静舒适的睡眠环境。\n\n### 4. **心理健康**\n   - **管理压力**：通过冥想、深呼吸、瑜伽或兴趣爱好放松心情。\n   - **社交支持**：与家人朋友保持联系，分享情绪，避免孤独感。\n   - **积极","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null,"reasoning_content":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":21,"total_tokens":533,"completion_tokens":512,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```