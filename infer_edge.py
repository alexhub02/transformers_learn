from transformers import AutoModelForCausalLM, AutoTokenizer
from models.glm_4_47.configuration_glm import GlmConfig
import os
import json
from models.glm_4_47.modeling_glm import GlmForCausalLM
import safetensors.torch

model_path = "/Users/qinggeiwolaipenfan/Desktop/proj/llm/chkpoint/chat-glm-1.5B-hf"

with open(os.path.join(model_path,"config.json"), "r", encoding="utf-8") as f:
    config_dict = json.load(f)
config = GlmConfig(**config_dict)
print(config)

model = GlmForCausalLM(config)
model_weight = safetensors.torch.load_file(os.path.join(model_path, "model.safetensors"))
model.load_state_dict(model_weight, strict=False)
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "who are you?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=500)
print(tokenizer.decode(outputs[0]))