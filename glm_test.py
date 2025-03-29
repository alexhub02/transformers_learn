from transformers import AutoModel, AutoTokenizer
import torch

# 1. 加载模型和分词器（支持本地路径或HuggingFace Hub）
model_path = "/Users/qinggeiwolaipenfan/Desktop/proj/llm/chkpoint/chat-glm-1.5B-hf"  # 其他选项："THUDM/chatglm2-6b", "THUDM/chatglm-6b"


from transformers import AutoTokenizer, GlmForCausalLM

model = GlmForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "who are you?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(result)