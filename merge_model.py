import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

lora_id = "tanamettpk/TC-instruct-DPO"
base_model_id = "qlora-out"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    device_map={"": 0},
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

merged_model= PeftModel.from_pretrained(base_model, lora_id)
merged_model= merged_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("merged_model", safe_serialization=True)
tokenizer.save_pretrained("merged_model")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
