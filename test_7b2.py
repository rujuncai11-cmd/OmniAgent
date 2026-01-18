"""
✅ 加载本地已下载的模型
- 识别并使用本地已下载的模型
- 不会重新下载
- 支持本地路径直接加载
"""

import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_local_model():
    """加载本地已下载的模型"""
    print("\n" + "="*70)
    print("📦 加载本地模型（Qwen2-7B GPTQ）")
    print("="*70)
    
    # 方法 1：直接使用本地路径（推荐！）
    # 这样可以100%确保加载本地模型，不会重新下载
    model_path = r"D:\HF_models\models--Qwen2.5-3B-Instruct-GPTQ-Int4\snapshots\main"
    
    print(f"\n📂 模型路径: {model_path}")
    
    # 验证路径是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在！")
        return None, None
    
    print(f"✓ 模型路径存在")
    
    # 列出模型文件
    print(f"\n📋 模型文件:")
    try:
        files = os.listdir(model_path)
        for f in sorted(files):
            full_path = os.path.join(model_path, f)
            if os.path.isfile(full_path):
                size_mb = os.path.getsize(full_path) / 1e6
                print(f"   - {f} ({size_mb:.1f}MB)")
            else:
                print(f"   - {f}/ (文件夹)")
    except Exception as e:
        print(f"   ✗ 无法列出文件: {e}")
    
    print(f"\n1️⃣  加载 tokenizer（从本地路径）...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,  # ← 直接用本地路径，不用模型 ID
            trust_remote_code=True,
            local_files_only=True  # ← 关键！只从本地加载
        )
        print("   ✓ tokenizer 加载成功")
    except Exception as e:
        print(f"   ✗ tokenizer 加载失败: {e}")
        return None, None
    
    print(f"\n2️⃣  加载模型（从本地路径）...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,  # ← 直接用本地路径，不用模型 ID
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True  # ← 关键！只从本地加载
        )
        print("   ✓ 模型加载成功")
    except Exception as e:
        print(f"   ✗ 模型加载失败: {e}")
        return None, None
    
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
    print("    Qwen2-7B GPTQ 本地模型")
    print("    加载本地已下载的模型（不会重新下载）")
    print("🚀 " + "="*66 + " 🚀")
    
    try:
        # 加载本地模型
        model, tokenizer = load_local_model()
        
        if model is None or tokenizer is None:
            print("\n❌ 模型加载失败，请检查路径是否正确")
            return
        
        # 显示模型信息
        print("\n" + "="*70)
        print("ℹ️  模型信息")
        print("="*70)
        print(f"模型: Qwen2-7B GPTQ 4bit")
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
        print("\n💡 使用本地模型的优点:")
        print("   - 不需要网络连接")
        print("   - 加载速度快")
        print("   - 不会重复下载")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()