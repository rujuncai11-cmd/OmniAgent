"""
✅ 稳定版：Qwen2.5-3B GPTQ 完整管道
已验证可在 GTX 1650 + CUDA 13.1 + transformers 4.45.2 上正常工作
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path):
    """加载 GPTQ 模型"""
    print("\n" + "="*70)
    print("📦 模型加载")
    print("="*70)
    
    print("\n1️⃣  加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    print("   ✓ tokenizer 加载成功")
    
    print("\n2️⃣  加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    print("   ✓ 模型加载成功")
    
    # 设置评估模式
    model.eval()
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_tokens=100):
    """生成文本"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    elapsed = time.time() - start_time
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
    
    return result, elapsed, tokens_generated


def main():
    """主程序"""
    
    print("\n" + "🚀 " + "="*66 + " 🚀")
    print("    Qwen2.5-3B GPTQ 文本生成系统")
    print("🚀 " + "="*66 + " 🚀")
    
    model_path = r"D:\HF_models\hub\models--Qwen2.5-3B-Instruct-GPTQ-Int4\snapshots\main"
    
    # 加载模型
    model, tokenizer = load_model(model_path)
    
    # 显示模型信息
    print("\n" + "="*70)
    print("ℹ️  模型信息")
    print("="*70)
    print(f"设备: {next(model.parameters()).device}")
    print(f"数据类型: {next(model.parameters()).dtype}")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU 显存: {allocated:.2f}GB / {total:.2f}GB")
    
    # 测试生成
    print("\n" + "="*70)
    print("📝 文本生成测试")
    print("="*70)
    
    test_prompts = [
        "AI的未来是",
        "机器学习的三个主要方向包括",
        "Python在数据科学中的应用",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}️⃣  提示词: '{prompt}'")
        print("-" * 70)
        
        result, elapsed, tokens = generate_text(
            model, tokenizer, prompt, max_tokens=80
        )
        
        print(f"生成结果:")
        print(result)
        print("-" * 70)
        print(f"⏱️  耗时: {elapsed:.2f}s | 📊 速度: {tokens/elapsed:.2f} tokens/s")
    
    print("\n" + "="*70)
    print("✅ 所有测试完成！")
    print("="*70)
    print("\n💡 模型已加载，可用于:")
    print("   - RAG (检索增强生成)")
    print("   - 微调 (LoRA)")
    print("   - API 部署 (FastAPI/Streamlit)")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()