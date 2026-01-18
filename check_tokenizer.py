from transformers import AutoTokenizer

model_path = r'D:\HF_models\hub\models--Qwen2.5-3B-Instruct-GPTQ-Int4\snapshots\main'

print("检查 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print(f'Tokenizer 类型: {type(tokenizer)}')
print(f'Vocab 大小: {len(tokenizer)}')

# 测试编码/解码
test_text = 'AI的未来是'
ids = tokenizer.encode(test_text)
decoded = tokenizer.decode(ids)

print(f'\n原文: {test_text}')
print(f'编码后: {ids}')
print(f'解码后: {decoded}')

# 如果解码后和原文一样，说明 tokenizer 正常
if decoded == test_text:
    print("\n✅ Tokenizer 正常")
else:
    print(f"\n❌ Tokenizer 有问题！解码结果不匹配")