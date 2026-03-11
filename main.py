
#############

## Imports ##

#############

import json
from datasets import Dataset
from datetime import datetime
from pathlib import Path

from helper_fcns import trim_text
from run_fcns import attack_texts, detect_with_methods, compute_metrics
from attack_class import Attack_Oracle
from detection_class import Detection_Oracle
from main_loop import run_attacks_and_compute_scores
from parse_data import parse_data_complete, collect_latex_text
from API_quality_check import API_quality_check, extract_best_models
from RADAR_considerations import RADAR_parse

with open("data/Full-RAID-NEWS.json", "r") as file:
    data = json.load(file)

number_of_texts = 50

# To make compatible with RADAR and RoBERTA, we reduce number of words
human_texts = [trim_text(data['human'][i]) for i in range(number_of_texts)];
ai_texts = [trim_text(data['ai'][i]) for i in range(number_of_texts)];
dipper_texts = [trim_text(data['dipper'][i]) for i in range(number_of_texts)];

today = datetime.today()
date_str = today.strftime("%y-%m-%d")
date_str = '26-02-27_SP_test' 
date_str = '26-02-28' 

specs = ['full text paraphrase', 'mask plus full text paraphrase', 'sentence paraphrase', 'mask plus sentence paraphrase']
# specs = ['full text paraphrase']

abbr_specs = [['full text paraphrase', 'FTP'], ['mask plus full text paraphrase', 'M+FTP'], ['sentence paraphrase', 'SP'], ['mask plus sentence paraphrase', 'M+SP'], ['None', 'None']]

methods = ['Fast-DetectGPT', 'RADAR', 'RoBERTa', 'LLMDet', 'Binoculars', 'DeTeCtive']

temps = [0.0, 0.5, 1.0, 1.5]
reps = [1.0, 1.2] # for mistral: [1.0, 1.6], for llama3: [1.0, 1.2]
masks = [15]


model_name = "llama3:instruct"
model_name = "mistral-small3.2:24b-instruct-2506-q4_K_M"
detection_oracle = Detection_Oracle()

run_attacks_and_compute_scores(model_name, detection_oracle, specs, methods, ai_texts, human_texts, temps, reps, masks, date_str)
parse_data_complete(date_str)
API_quality_check(date_str)
extract_best_models(date_str, abbr_specs)
collect_latex_text(date_str)


## Looking at the data we decide to go ahead with mistral FTP t = 1.5 rp 1.0, SP t=1.5, rp= 1.0 and 
## llama3 FTP t = 0.5, rp = 1.2 SP t = 1.0, rp = 1.2

# total data is 1780
number_of_texts = 500  

# To make compatible with RADAR and RoBERTA, we reduce number of words
human_texts = [trim_text(data['human'][i]) for i in range(number_of_texts)];
ai_texts = [trim_text(data['ai'][i]) for i in range(number_of_texts)];
dipper_texts = [trim_text(data['dipper'][i]) for i in range(number_of_texts)];

date_str = '26-02-28_500' 

specs = ['full text paraphrase']
abbr_specs = [['full text paraphrase', 'FTP'], ['mask plus full text paraphrase', 'M+FTP'], ['sentence paraphrase', 'SP'], ['mask plus sentence paraphrase', 'M+SP'], ['None', 'None']]
methods = ['Fast-DetectGPT', 'RADAR', 'RoBERTa', 'LLMDet', 'Binoculars', 'DeTeCtive']

temps = [1.5]
reps = [1.0] # for mistral: [1.0, 1.6], for llama3: [1.0, 1.2]
masks = [15]

model_name = "mistral-small3.2:24b-instruct-2506-q4_K_M"
detection_oracle = Detection_Oracle()

run_attacks_and_compute_scores(model_name, detection_oracle, specs, methods, ai_texts, human_texts, temps, reps, masks, date_str)


specs = ['sentence paraphrase']

model_name = "mistral-small3.2:24b-instruct-2506-q4_K_M"
detection_oracle = Detection_Oracle()

run_attacks_and_compute_scores(model_name, detection_oracle, specs, methods, ai_texts, human_texts, temps, reps, masks, date_str)



model_name = "mistral-small3.2:24b-instruct-2506-q4_K_M"

temps = [1.5]
reps = [1.0] # for mistral: [1.0, 1.6], for llama3: [1.0, 1.2]
masks = [15]

specs = ['full text paraphrase', 'sentence paraphrase']
specs = ['full text paraphrase']
run_attacks_and_compute_scores(model_name, detection_oracle, specs, methods, ai_texts, human_texts, temps, reps, masks, date_str)


model_name = "llama3:instruct"

temps = [0.5]
reps = [1.2] # for mistral: [1.0, 1.6], for llama3: [1.0, 1.2]
masks = [15]

specs = ['full text paraphrase']
run_attacks_and_compute_scores(model_name, detection_oracle, specs, methods, ai_texts, human_texts, temps, reps, masks, date_str)

model_name = "llama3:instruct"

temps = [1.0]
reps = [1.2] # for mistral: [1.0, 1.6], for llama3: [1.0, 1.2]
masks = [15]

specs = ['sentence paraphrase']
run_attacks_and_compute_scores(model_name, detection_oracle, specs, methods, ai_texts, human_texts, temps, reps, masks, date_str)

###################

parse_data_complete(date_str)
API_quality_check(date_str)
extract_best_models(date_str, abbr_specs)
collect_latex_text(date_str)

####################################################################################
####################################################################################
####################################################################################

# To make compatible with RADAR and RoBERTA, we reduce number of words
human_texts = [trim_text(data['human'][i]) for i in range(number_of_texts)];
ai_texts = [trim_text(data['ai'][i]) for i in range(number_of_texts)];
dipper_texts = [trim_text(data['dipper'][i]) for i in range(number_of_texts)];

today = datetime.today()
date_str = today.strftime("%y-%m-%d")
date_str = '26-02-23' 

specs = ['full text paraphrase', 'mask plus full text paraphrase', 'sentence paraphrase', 'mask plus sentence paraphrase']
abbr_specs = [['full text paraphrase', 'FTP'], ['mask plus full text paraphrase', 'M+FTP'], ['sentence paraphrase', 'SP'], ['mask plus sentence paraphrase', 'M+SP'], ['None', 'None']]

methods = ['Fast-DetectGPT', 'RADAR', 'RoBERTa', 'LLMDet', 'Binoculars', 'DeTeCtive']

temps = [0.0, 0.5, 1.0, 1.5]
reps = [1.0, 1.2] # for mistral: [1.0, 1.6], for llama3: [1.0, 1.2]
masks = [15]

model_name = "llama3:instruct"
model_name = "mistral-small3.2:24b-instruct-2506-q4_K_M"
detection_oracle = Detection_Oracle()