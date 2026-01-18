"""
✅ Qwen2.5-3B FP16 官方版本 - 显示模型路径
- 自动下载官方模型
- 显示模型所在的实际路径
- 显存占用：4GB
- 完全无乱码
"""

import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

def find_model_path(model_id, cache_dir):
    """查找模型的实际路径"""
    print(f"\n🔍 查找模型路径...")
    
    # 获取 HF 默认缓存位置
    from huggingface_hub import HfApi
    api = HfApi()
    
    # 模型 ID 转换为文件夹名
    # "Qwen/Qwen2.5-3B-Instruct" -> "models--Qwen--Qwen2.5-3B-Instruct"
    repo_name = model_id.replace("/", "--")
    model_folder_name = f"models--{repo_name}"
    
    # 检查可能的路径
    possible_paths = [
        os.path.join(cache_dir, "hub", model_folder_name),  # 指定的 cache_dir
        os.path.join(cache_dir, model_folder_name),          # cache_dir 直接下
        os.path.expanduser(f"~/.cache/huggingface/hub/{model_folder_name}"),  # 默认位置
    ]
    
    print(f"   检查以下路径:")
    for path in possible_paths:
        print(f"   - {path}")
        if os.path.exists(path):
            print(f"     ✓ 找到!")
            return path
    
    print(f"   ✗ 未在上述位置找到")
    
    # 如果都找不到，尝试下载/检查
    try:
        print(f"\n   尝试使用 snapshot_download 查询...")
        actual_path = snapshot_download(
            model_id,
            cache_dir=cache_dir,
            resume_download=True
        )
        print(f"   ✓ 模型路径: {actual_path}")
        return actual_path
    except Exception as e:
        print(f"   ✗ 查询失败: {e}")
        return None


def load_model():
    """加载 FP16 官方版本"""
    print("\n" + "="*70)
    print("📦 模型加载（FP16 官方版本）")
    print("="*70)
    
    # 使用官方未量化的 FP16 版本
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    cache_dir = r"D:\HF_models"
    
    print(f"\n📍 配置信息:")
    print(f"   模型 ID: {model_id}")
    print(f"   缓存目录: {cache_dir}")
    
    # 查找模型路径
    model_path = find_model_path(model_id, cache_dir)
    
    print(f"\n1️⃣  加载 tokenizer...")
    print(f"   （首次会从 HuggingFace 下载，约 1GB）")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    print("   ✓ tokenizer 加载成功")
    
    print(f"\n2️⃣  加载模型...")
    print(f"   （首次会从 HuggingFace 下载，约 6GB）")
    print(f"   请耐心等待...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    print("   ✓ 模型加载成功")
    
    # 再次查询（确保找到）
    model_path = find_model_path(model_id, cache_dir)
    
    model.eval()
    
    return model, tokenizer, model_path


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
    print("    Qwen2.5-3B FP16 官方版本")
    print("    自动下载 + 显示路径 + 生成测试")
    print("🚀 " + "="*66 + " 🚀")
    
    try:
        # 加载模型（首次会自动下载）
        model, tokenizer, model_path = load_model()
        
        # 显示模型信息
        print("\n" + "="*70)
        print("ℹ️  模型信息")
        print("="*70)
        print(f"版本: FP16 官方版本（未量化）")
        print(f"设备: {next(model.parameters()).device}")
        print(f"数据类型: {next(model.parameters()).dtype}")
        
        if model_path:
            print(f"\n📂 模型路径:")
            print(f"   {model_path}")
            
            # 显示模型文件
            if os.path.exists(model_path):
                print(f"\n📋 模型文件:")
                try:
                    files = os.listdir(model_path)
                    for f in sorted(files)[:10]:  # 只显示前 10 个文件
                        full_path = os.path.join(model_path, f)
                        if os.path.isfile(full_path):
                            size_mb = os.path.getsize(full_path) / 1e6
                            print(f"   - {f} ({size_mb:.1f}MB)")
                        else:
                            print(f"   - {f}/ (文件夹)")
                    if len(files) > 10:
                        print(f"   ... 还有 {len(files) - 10} 个文件")
                except Exception as e:
                    print(f"   无法列出文件: {e}")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n💾 GPU 显存: {allocated:.2f}GB / {total:.2f}GB")
        
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
        print("\n下次运行会更快（不需要重新下载）")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 如果是网络问题，尝试：")
        print("   1. 检查网络连接")
        print("   2. 确保能访问 huggingface.co")
        print("   3. 检查磁盘空间（需要 15GB 以上）")


if __name__ == "__main__":
    main()