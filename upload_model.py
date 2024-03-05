import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
import time

base_model_id = "merged_model"


input_text = """
### Instruction:
จงสร้าง persona ของใครก็ได้ 1 คน

### Response:
"""

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    device_map={"": 0},
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.5,
    max_new_tokens=300,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id)

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Generate outputs
st_time = time.time()
outputs = model.generate(**inputs, generation_config=generation_config)

# Decode and print response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response time: {time.time() - st_time} seconds")

print(response)

isPush = input("want to push?[y/n] :")
if isPush == "y":
    model.push_to_hub("tanamettpk/TC-novid-v1")
