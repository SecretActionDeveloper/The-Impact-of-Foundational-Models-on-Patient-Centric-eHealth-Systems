# Step 1 feature exctraction using fine tuned model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import pandas as pd
from transformers import GenerationConfig
from time import perf_counter
import re
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

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
  return(tokenizer.decode(outputs[0], skip_special_tokens=True))

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

model_id="Artemis13/llama3.18B-Feature_extractor"
model, tokenizer = get_model_and_tokenizer(model_id)
import pandas as pd
path = "./all_apps_cleaned_df.csv"
df = pd.read_csv(path)

prompt = '''Extract the features of the application from the description and output them as python list. The features should be short phrases (10 words or fewer) that describe the functionalities of the app.
App description: [/INST]'''

# Initialize the column before filling
df["Feature_set"] = ""

# Use tqdm for progress bar
for i in tqdm(range(len(df)), desc="Extracting features"):
    des = df["description"][i]
    response = generate_response(user_input=prompt + des)
    features=[]
    match = re.search(r"assistant:.*?\n(\[.*?\])<\|im_end\|>", response, re.DOTALL)
    if match:
        features_str = match.group(1)
        try:
            features = eval(features_str)
        except:
          pass

    df["Feature_set"][i]= features

df.to_csv("./Application_with_features.csv", index=False)