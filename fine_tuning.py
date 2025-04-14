
from datasets import  Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os
from transformers import GenerationConfig
from time import perf_counter

import pandas as pd
def prepare_train_data(data):
    # Convert the data to a Pandas DataFrame
    data_df = pd.DataFrame(data)
    # Create a new column called "text"
    data_df["text"] = data_df[["prompt", "response"]].apply(lambda x: "<|im_start|>user\n" + x["prompt"] + " <|im_end|>\n<|im_start|>assistant\n" + x["response"] + "<|im_end|>\n", axis=1)
    # Create a new Dataset from the DataFrame
    data = Dataset.from_pandas(data_df)
    return data

def get_model_and_tokenizer(model_id):
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.pad_token = tokenizer.eos_token
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
  )
  model = AutoModelForCausalLM.from_pretrained(
      model_id, quantization_config=bnb_config, device_map="auto"
  )
  model.config.use_cache=False
  model.config.pretraining_tp=1
  return model, tokenizer

def formatted_prompt(question)-> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant:"
    
def generate_response(user_input):
  prompt = formatted_prompt(user_input)
  inputs = tokenizer([prompt], return_tensors="pt")
  generation_config = GenerationConfig(penalty_alpha=0.6,do_sample = True,
      top_k=5,temperature=0.5,repetition_penalty=1.2,
      max_new_tokens=60,pad_token_id=tokenizer.eos_token_id
  )
  start_time = perf_counter()
  inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
  outputs = model.generate(**inputs, generation_config=generation_config)
  theresponse = (tokenizer.decode(outputs[0], skip_special_tokens=True))
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))
  output_time = perf_counter() - start_time
  print(f"Time taken for inference: {round(output_time,2)} seconds")
  
def formatted_train(input,response)->str:
    return f"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"
  
model_id="meta-llama/Meta-Llama-3.1-8B"
model, tokenizer = get_model_and_tokenizer(model_id)

output_model="llama3.18B-Feature_extractor"

df = pd.read_csv("fine_tuning_df.csv")
cols_to_keep = ['appId','description', 'uncleaned_LLM_features', 'Correct_feature']
# Keep only those columns
df = df[cols_to_keep]
df = df[df['Correct_feature']]

training_data = []
apps = list(df['appId'].unique())

prompt = '''Extract the features of the application from the description and output them as python list. The features should be short phrases (10 words or fewer) that describe the functionalities of the app.
App description: [/INST]'''


for app in apps:
  app_df = df[df['appId'] == app].reset_index(drop=False)
  des = app_df['description'][0]
  features = list(app_df['uncleaned_LLM_features'])
  training_data.append({"prompt": prompt + des, "response": str(features)})

data = prepare_train_data(training_data)

peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

training_arguments = TrainingArguments(
    output_dir=output_model,
    per_device_train_batch_size=1,        
    gradient_accumulation_steps=4,       
    optim="paged_adamw_32bit",
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",                
    save_total_limit=1,                   
    logging_steps=5,
    num_train_epochs=3,                  
    max_steps=100,                        
    fp16=True,
    push_to_hub=True                   
)
trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=peft_config,
        dataset_text_field="text",
        args=training_arguments,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=1024
    )

trainer.train()
