import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r'D:\HF_models\hub\models--Qwen2.5-3B-Instruct-GPTQ-Int4\snapshots\main'

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("✓ tokenizer 加载成功")

print("\n加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map='auto', 
    trust_remote_code=True
)
print("✅ GPTQ 模型加载成功！")

# 简单生成测试
print("\n生成测试...")
prompt = "AI的未来是"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"结果: {result}")