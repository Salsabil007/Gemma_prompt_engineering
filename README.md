## Prompt Engineering on Gemma for python Q/A
In this repository, I implement different prompt engineering techniques on the Google's recently launched Gemma model to answer common python questions. I implemented several approaches including finetuning gemma using LORA, few-shot prompting, 
and zero-shot chain of thought to enable efficient Q/A.

I used the Gemma 2B parameter instruction trained version built using the Huggingface transformers for this task. 

### Finetuning using LORA
Before we begin, we need to request access to the gemma model from huggingface as it is a gated model. To do so, follow the link https://huggingface.co/google/gemma-7b and submit the consent form in Terms. Once you get access to the model, 
go to the huggingface profile -> settings -> Access Tokens and generate a new token, preferably in READ form. In your google colab, click on the Sectets (key submol) tab in the left sidebar and add your token and use "HF_TOKEN" as name. 
In google colab, go to connect -> change runtime type -> Connect to T4 GPU.

We will use parameter efficient finetuning technique (PEFT) LORA to finetune the gemma model. Generally, finetuning a LLM with 2 Billion parameters is complex and resource as well as time consuming. LoRA or low rank adaptation enables us to
train only a fraction of the weights in the LLM while freezing the other weights. LoRA leverages the rank property of a matrix. Lower rank matrix are useful in data compression as it compress the size of a matrix while preserving sufficient 
information. More details can be found in this [link](https://www.datacamp.com/tutorial/mastering-low-rank-adaptation-lora-enhancing-large-language-models-for-efficient-adaptation). In short, we can use LoRA to train only a small subset of 
parameters or low-rank matrices associated with the layers we specify. We can select the layers that we want to decompose into low-rank matrix called LoRA adapter and train the adapters and add to the existing model during prediction.


# Reference
[1] https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fwww.datacamp.com%2Ftutorial%2Fmastering-low-rank-adaptation-lora-enhancing-large-language-models-for-efficient-adaptation

[2] https://huggingface.co/blog/gemma-peft

[3] https://www.datacamp.com/tutorial/mastering-low-rank-adaptation-lora-enhancing-large-language-models-for-efficient-adaptation

[4] Paul Mooney, Ashley Chow. (2024). Google – AI Assistants for Data Tasks with Gemma. Kaggle. https://kaggle.com/competitions/data-assistants-with-gemma
