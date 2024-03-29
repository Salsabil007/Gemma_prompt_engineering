

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import BitsAndBytesConfig
import os
from datasets import load_dataset

#from transformers import BitsAndBytesConfig, GemmaTokenizer

model_id = "google/gemma-2b-it"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
#quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

text = "Human: What is lambda expression? Assistant:"
device = "cuda"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

data = load_dataset("KonradSzafer/stackoverflow_python_preprocessed")
data = data["train"].train_test_split(test_size=0.2)

#data = data.map(lambda samples: tokenizer(samples["question"]), batched=True)
#print(data)

train_set = data['train']
test_set = data['test']

"""### Zero shot Chain of Thought

In the subsequent sections, we apply zero-shot chain of thought approach described in this paper https://arxiv.org/pdf/2205.11916.pdf
"""

def formatting_func2(question):
    text = f"Question: {question}\nAnswer: Let's think step-by-step.\n"
    return text
print(formatting_func2("What is your name?"))

device = "cuda"
input_text = formatting_func2(test_set['question'][1])
input_ids = tokenizer(input_text, return_tensors="pt").to(device)
outputs = model.generate(**input_ids, max_new_tokens=256)
print(tokenizer.decode(outputs[0],skip_special_tokens=True))

"""### Zero shot prompting"""

def formatting_func(question):
    text = f"Question: {question}\nAnswer: The answer is\n"
    return text
print(formatting_func("What is your name?"))

device = "cuda"
input_text = formatting_func(test_set['question'][1])
input_ids = tokenizer(input_text, return_tensors="pt").to(device)
outputs = model.generate(**input_ids, max_new_tokens=256)
print(tokenizer.decode(outputs[0],skip_special_tokens=True))

