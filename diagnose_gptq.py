"""
GPTQ 修复验证脚本
逐步诊断和修复 GPTQ 加载问题
"""

import os
import sys
import subprocess

def check_versions():
    """检查关键库版本"""
    print("\n" + "="*70)
    print("1️⃣  版本检查")
    print("="*70)
    
    packages = {
        'torch': '需要 2.0+',
        'transformers': '需要 4.45.2（或 4.45.x）',
        'auto-gptq': '需要 0.7.1+',
        'optimum': '需要 1.17.0+',
    }
    
    import torch
    from transformers import __version__ as transformers_version
    
    print(f"\n✓ torch: {torch.__version__}")
    print(f"  建议: 2.0+，你的可以")
    
    print(f"\n✓ transformers: {transformers_version}")
    if transformers_version.startswith(('4.45', '4.46', '4.47', '4.48')):
        print(f"  ✅ 版本 OK（稳定版本）")
    elif transformers_version.startswith('4.57'):
        print(f"  ⚠️  版本过新（已知有 GPTQ bug）")
        print(f"  💡 建议降级到 4.45.2")
        print(f"\n  运行这条命令降级：")
        print(f"  pip install transformers==4.45.2")
    else:
        print(f"  ⚠️  版本未测试")
    
    try:
        import auto_gptq
        print(f"\n✓ auto-gptq: {auto_gptq.__version__}")
    except ImportError:
        print(f"\n❌ auto-gptq: 未安装")
    
    try:
        import optimum
        print(f"✓ optimum: {optimum.__version__}")
    except ImportError:
        print(f"❌ optimum: 未安装")


def check_env_vars():
    """检查环境变量"""
    print("\n" + "="*70)
    print("2️⃣  环境变量检查")
    print("="*70)
    
    env_vars = {
        'DISABLE_EXLLAMA': '应该是 1',
        'DISABLE_EXLLAMAV2': '应该是 1',
        'EXLLAMA_NO_CUDA_EXTENSION': '应该是 1',
    }
    
    for var_name, expected in env_vars.items():
        value = os.environ.get(var_name, '未设置')
        status = "✓" if value == '1' else "⚠️"
        print(f"{status} {var_name}: {value}")
    
    if os.environ.get('DISABLE_EXLLAMA') != '1':
        print(f"\n💡 环境变量未正确设置")
        print(f"  运行这条命令（需要管理员）：")
        print(f"  conda env config vars set DISABLE_EXLLAMA=1")
        print(f"  conda env config vars set DISABLE_EXLLAMAV2=1")
        print(f"  然后重新激活环境")


def check_model_files():
    """检查模型文件"""
    print("\n" + "="*70)
    print("3️⃣  模型文件检查")
    print("="*70)
    
    model_path = r"D:\HF_models\hub\models--Qwen2.5-3B-Instruct-GPTQ-Int4\snapshots\main"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    print(f"✓ 模型目录存在")
    
    files = os.listdir(model_path)
    required = ['config.json', 'tokenizer.json']
    weights = [f for f in files if f.endswith(('.safetensors', '.bin'))]
    
    print(f"\n主要文件：")
    for req in required:
        if req in files:
            size = os.path.getsize(os.path.join(model_path, req)) / 1024
            print(f"  ✓ {req} ({size:.0f}KB)")
        else:
            print(f"  ❌ {req} (缺失)")
    
    if weights:
        for w in weights:
            size = os.path.getsize(os.path.join(model_path, w)) / (1024**3)
            print(f"  ✓ {w} ({size:.1f}GB)")
    else:
        print(f"  ❌ 模型权重文件缺失！")
        return False
    
    return True


def test_gptq_load():
    """测试 GPTQ 加载"""
    print("\n" + "="*70)
    print("4️⃣  GPTQ 加载测试")
    print("="*70)
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_path = r"D:\HF_models\hub\models--Qwen2.5-3B-Instruct-GPTQ-Int4\snapshots\main"
        
        print(f"\n加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✓ tokenizer 加载成功")
        
        print(f"\n加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        print(f"✓ 模型加载成功！")
        
        # 简单生成测试
        print(f"\n生成测试...")
        prompt = "AI的未来是"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ 生成成功！")
        print(f"\n结果: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载失败: {str(e)[:200]}")
        return False


def suggest_fix():
    """建议修复"""
    print("\n" + "="*70)
    print("💡 修复建议")
    print("="*70)
    
    print("""
根据上面的诊断，这是修复步骤：

【如果 transformers 版本是 4.57.x】
→ 最可能的原因就是版本 bug
→ 运行这条命令降级：
   pip install transformers==4.45.2

【如果环境变量未正确设置】
→ 在 Anaconda Prompt 中运行：
   conda activate omniagent
   conda env config vars set DISABLE_EXLLAMA=1
   conda env config vars set DISABLE_EXLLAMAV2=1
   conda deactivate
   conda activate omniagent

【如果模型文件缺失】
→ 重新下载：
   huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 \\
     --local-dir D:\\HF_models\\hub\\models--Qwen2.5-3B-Instruct-GPTQ-Int4

【完成后，重新运行这个诊断脚本验证】
    """)


def main():
    print("\n🔍 GPTQ 诊断 & 修复工具")
    print("="*70)
    
    check_versions()
    check_env_vars()
    files_ok = check_model_files()
    
    if not files_ok:
        print("\n❌ 模型文件有问题，无法继续测试")
        suggest_fix()
        return
    
    success = test_gptq_load()
    
    if success:
        print("\n" + "="*70)
        print("✅ GPTQ 工作正常！可以继续开发")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ GPTQ 加载失败")
        print("="*70)
        suggest_fix()


if __name__ == "__main__":
    main()