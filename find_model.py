import os

print("\n" + "="*70)
print("🔍 查找 HuggingFace 模型位置")
print("="*70)

# 获取 HF 缓存位置
hf_home = os.environ.get('HF_HOME')
print(f"\n1️⃣  环境变量检查:")
if hf_home:
    print(f"   HF_HOME 设置为: {hf_home}")
else:
    print(f"   HF_HOME 未设置，使用默认位置")

# 检查常见路径
print(f"\n2️⃣  检查常见缓存路径:\n")

common_paths = [
    r"C:\Users\CRJ\.cache\huggingface\hub",
    r"D:\HF_models",
    r"D:\HF_models\hub",
]

found_models = []

for path in common_paths:
    if os.path.exists(path):
        print(f"✓ 路径存在: {path}")
        # 列出该目录下的所有文件夹
        try:
            contents = os.listdir(path)
            qwen_dirs = [d for d in contents if "Qwen" in d and os.path.isdir(os.path.join(path, d))]
            
            if qwen_dirs:
                print(f"  ✓ 找到 {len(qwen_dirs)} 个 Qwen 模型:")
                for d in qwen_dirs:
                    full_path = os.path.join(path, d)
                    try:
                        # 计算文件夹大小
                        total_size = 0
                        for dirpath, dirnames, filenames in os.walk(full_path):
                            for filename in filenames:
                                total_size += os.path.getsize(os.path.join(dirpath, filename))
                        size_gb = total_size / 1e9
                        print(f"    ├─ {d}")
                        print(f"    │  └─ 大小: {size_gb:.2f}GB")
                        print(f"    │  └─ 路径: {full_path}")
                        found_models.append((d, full_path, size_gb))
                    except Exception as e:
                        print(f"    └─ {d} (无法计算大小)")
            else:
                print(f"  ✗ 未找到 Qwen 相关模型")
        except Exception as e:
            print(f"  ✗ 无法访问: {e}")
    else:
        print(f"✗ 路径不存在: {path}")
    
    print()

# 总结
print("="*70)
print("📊 查询结果总结")
print("="*70)

if found_models:
    print(f"\n✅ 找到 {len(found_models)} 个模型:\n")
    for name, path, size in found_models:
        print(f"模型: {name}")
        print(f"大小: {size:.2f}GB")
        print(f"路径: {path}\n")
else:
    print("\n❌ 未找到任何 Qwen 模型")
    print("\n💡 可能的原因:")
    print("   1. 模型还未下载")
    print("   2. 模型在其他路径")
    print("   3. 模型文件夹名称不同\n")

# 检查磁盘空间
print("="*70)
print("💾 磁盘空间检查")
print("="*70 + "\n")

import shutil
try:
    d_drive = shutil.disk_usage("D:\\")
    total_gb = d_drive.total / 1e9
    used_gb = d_drive.used / 1e9
    free_gb = d_drive.free / 1e9
    print(f"D 盘:")
    print(f"  总空间: {total_gb:.2f}GB")
    print(f"  已用: {used_gb:.2f}GB")
    print(f"  可用: {free_gb:.2f}GB")
except:
    pass