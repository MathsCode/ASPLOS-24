from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from typing import Optional
import json
import time 
def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions
question_list = load_questions('/home/xujiaming/xujiaming/research/ASPLOS-24/EAGLE/eagle/data/mt_bench/question.jsonl',begin=0,end=80)

# /share/datasets/public_models/Llama-2-7b-chat-hf
# /home/xujiaming/xujiaming/datasets/LLM_models/EAGLE-llama2-chat-7B
# /home/xujiaming/xujiaming/datasets/LLM_models/EAGLE-Vicuna-33B-v1.3
# /share/datasets/public_models/lmsys_vicuna-33b-v1.3
model_name = "Llama2-13B"
model = EaModel.from_pretrained(
    base_model_path='/share/datasets/public_models/Llama-2-13b-chat-hf',
    ea_model_path="/home/xujiaming/xujiaming/datasets/LLM_models/EAGLE-llama2-chat-13B",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    # is_offload = False,
    # skip_model = "/home/xujiaming/xujiaming/research/ASPLOS-24/skip_layer/model.txt",
)
model.eval()
from tqdm import trange



exe_layers = [0,0,0,0,0]
exe_tokens = [0,0,0,0,0]
train_hidden_states = []
train_data = []
train_label = []
print(model.config.num_hidden_layers)
max_value_dist = [[] for i in range(model.config.num_hidden_layers)]

message = "What is the capital of France?"
# message = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
# message = "Hello"
# conv = get_conversation_template("llama-2-chat")  
# sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
# conv.system_message = sys_p
# conv.append_message(conv.roles[0], message)
# conv.append_message(conv.roles[1], None)
# prompt = conv.get_prompt() + " "

# input_ids=model.tokenizer([prompt]).input_ids
# input_ids = torch.as_tensor(input_ids).cuda()

# exit_layer_id_list=[]
# warm_up = 0
# repeat_iter = 1
# for _ in range(warm_up):
#     output_ids=model.eagenerate(input_ids,temperature=0,max_new_tokens=256,skip_layer=True,Early_Exiting=False,time_breakdown=False)
# st = time.time()
# for _ in range(repeat_iter):
#     output_ids=model.eagenerate(input_ids,temperature=0,max_new_tokens=256,skip_layer=False,Early_Exiting=True,time_breakdown=False,exe_layers=exe_layers,exe_tokens=exe_tokens,max_value_dist=max_value_dist,train_data=train_data,train_label=train_label,train_hidden_states=train_hidden_states)
# ed = time.time()
# print('time w/o. branch prediction:', (ed -st)/repeat_iter)
# output=model.tokenizer.decode(output_ids[0])
# print(output)
# train_hidden_states = torch.cat(train_hidden_states,dim=0)
# print(train_hidden_states.shape)
# train_data = torch.cat(train_data,dim=0)
# print(train_data.shape)
# train_data = torch.cat([train_hidden_states,train_data],dim=1)
# print(train_data.shape)
# train_label = torch.cat(train_label)
# print(train_label.shape)

# torch.save(train_data,'../results/train_data.pt')
# torch.save(train_label,'../results/train_label_new.pt')
# print(len(train_data))
# print(len(train_label))
# print(len(train_hidden_states))
# print('average:',[exe_layers[i]/exe_tokens[i]  for i in range(len(exe_layers))])


    

# try:
st = time.time()
for i in trange(len(question_list)):
    exe_layers = [0,0,0,0,0]
    exe_tokens = [0,0,0,0,0]
    train_hidden_states = []
    train_data = []
    train_label = []
    print("===================== Question Id = ",question_list[i]['question_id']," ======================")
    # message = "What is the capital of France?"
    if question_list[i]['question_id'] == 110 or question_list[i]['question_id'] == '110':
        continue
    message = question_list[i]['turns'][0]
    # message = "Hello"
    conv = get_conversation_template("llama-2-chat")  
    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p
    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " "

    input_ids=model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    output_ids=model.eagenerate(input_ids,temperature=0,max_new_tokens=256,skip_layer=False,Early_Exiting=True,time_breakdown=False,exe_layers=exe_layers,exe_tokens=exe_tokens,max_value_dist=max_value_dist,train_data=train_data,train_label=train_label,train_hidden_states=train_hidden_states)
    output=model.tokenizer.decode(output_ids[0])
    
    train_hidden_states = torch.cat(train_hidden_states,dim=0)
    train_data = torch.cat(train_data,dim=0)
    train_data_soft = torch.nn.functional.softmax(train_data,dim=-1)
    feature = torch.cat([train_hidden_states,train_data,train_data_soft],dim=-1)
    print(feature.shape)
    train_label = torch.cat(train_label)
    print(train_label.shape)

    # print(exe_layers)
    # print(exe_tokens)
    # print('average:',[exe_layers[i]/exe_tokens[i]  for i in range(len(exe_layers))])
    # [21.3, 21.17, 21.5, 21.3, 21.2]
    file_path_feature = '../results/'+model_name+'_feature.pt'
    file_path_label = '../results/'+model_name+'_label.pt'
    if os.path.exists(file_path_feature):
        feature_pre = torch.load(file_path_feature)
        feature = torch.cat([feature_pre,feature],dim=0)
        lable_pre = torch.load(file_path_label)
        train_label = torch.cat([lable_pre,train_label])
    torch.save(feature,file_path_feature)
    torch.save(train_label,file_path_label)
    # print(output)
ed = time.time()
print('MT-bench Time:', (ed -st))

print(train_hidden_states.shape)





# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# for i in range(len(max_value_dist)):
#     data = max_value_dist[i]
#     print(i,min(data))
#     weights = np.ones_like(data)/float(len(data))
#     data_series = pd.Series(data)
#     plt.figure(figsize=(14,7))
#     plt.hist(data_series,weights=weights)
#     plt.title('Histogram with KDE')
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.savefig("../results/"+str(i)+"_hist.png")



