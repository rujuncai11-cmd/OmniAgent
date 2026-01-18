import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("正在加载本地 Qwen2-7B-Instruct GPTQ 4bit 模型...")

# 本地模型路径（你已经下载好的）
model_name = r"D:\HF_models\hub\models--Qwen--Qwen2-7B-Instruct-GPTQ-Int4\snapshots\main"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 加载模型（GPTQ 模型最简单方式）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",          # 自动把能放GPU的放GPU，放不下的自动offload到CPU
    trust_remote_code=True,
    #local_files_only=True       # 只用本地文件，不联网
)

# 测试生成
prompt = "AI的未来是"

print(f"\n开始生成，提示词：'{prompt}'")
start_time = time.time()

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,        # 可以改大一点，GPTQ版速度快
    do_sample=True,
    temperature=0.7,
    top_p=0.8
)

end_time = time.time()
inference_time = end_time - start_time

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "="*60)
print("7B GPTQ 模型生成结果：")
print(result)
print("="*60)
print(f"总耗时：{inference_time:.2f} 秒")
print(f"生成速度约：{100 / inference_time:.2f} tokens/秒")
print("="*60)