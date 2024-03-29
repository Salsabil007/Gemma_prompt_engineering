import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import BitsAndBytesConfig
import os
from datasets import load_dataset


model_id = "google/gemma-2b-it"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
#quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)



data = load_dataset("KonradSzafer/stackoverflow_python_preprocessed")
data = data["train"].train_test_split(test_size=0.2)
print(data)

train_set = data['train']
test_set = data['test']

few_shot_examples = [
    ("Qestion:" + test_set['question'][0] + " : Answer: "+test_set['answer'][0]),
    ("Question: "+test_set['question'][1] + " : Answer: "+test_set['answer'][1]),
    ("Question: "+test_set['question'][2] + " : Answer: "+test_set['answer'][2]),
    ("Question: "+test_set['question'][3] + " : Answer: "+test_set['answer'][3]),
    #("Question: "+test_set['question'][4] + " : Answer: "+test_set['answer'][4]),
    #("Question: "+test_set['question'][5] + " : Answer: "+test_set['answer'][5]),
    #("Question: "+test_set['question'][6] + " : Answer: "+test_set['answer'][6]),
    #("Question: "+test_set['question'][20] + " : Answer: "+test_set['answer'][20]),
    #("Question: "+test_set['question'][30] + " : Answer: "+test_set['answer'][30]),
    #("Question: "+test_set['question'][40] + " : Answer: "+test_set['answer'][40]),
]


your_question = test_set['question'][4]
#prompt = "See the question answer examples below: \n".join(few_shot_examples) + "Answer the question below following the patterns of previous examples: \n" + your_question
prompt = "\n".join(few_shot_examples) + "\nAnswer the following question: " + your_question


input_ids = tokenizer(prompt, return_tensors="pt")
output = model.generate(**input_ids, max_new_tokens=256)
answer = tokenizer.decode(output[0], skip_special_tokens=True)
#f = answer.split("Answer the question below following the patterns of previous examples:")[-1]
f = answer.split("Answer the following question: ")[-1]
print(f)




