import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import os
from trl import SFTTrainer


model_id = "google/gemma-2b-it"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)


print(model)

#Test the pretrained based model on sample questions
text = "Human: Write a python program to remove duplicate elements from a list. Assistant:"
device = "cuda"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


text = "Human: Write a function to check if a given number is even or odd. Assistant:"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

#Finetuning using LORA
##we can see the layers of the model and select which layers we want to apply adapter weights.
##1_proj and v_proj are the attention layers.
##r -> the rank of the low rank matrices learned while finetuning. Larger r means a large number of parameters to be trained.
from peft import LoraConfig
lora_config = LoraConfig(
    r=8, #the rank of the low rank matrix
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], ##all the linear layers
    task_type="CAUSAL_LM",
)


from datasets import load_dataset

data = load_dataset("KonradSzafer/stackoverflow_python_preprocessed")

#data = data.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
#data.train_test_split(test_size=0.2)
#data = data["train"].train_test_split(test_size=0.2)
data = data.map(lambda samples: tokenizer(samples["question"]), batched=True)
print(data)


def formatting_func(example):
    text = f"Question: {example['question'][0]}\nanswer: {example['answer'][0]}"
    return [text]

trainer = SFTTrainer(
    model=model,
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=20,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)
trainer.train()


#Test the finetuned model on example questions
text = "Human: Write a python program to remove duplicate elements from a list. Assistant:"
device = "cuda"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))