from transformers import AutoTokenizer,AutoModelForCausalLM


from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from typing import Optional
import json
import time 
from tqdm import trange

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
# /share/datasets/public_models/lmsys_vicuna-33b-v1.3


if torch.cuda.is_available():
    device = torch.device("cuda:{}".format(0))
else:
    device = torch.device("cpu")


model_path = '/share/datasets/public_models/Llama-2-7b-chat-hf'
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

output_ids_tot = 0
st = time.time()
for i in trange(len(question_list)):
    torch.cuda.empty_cache()
    # print("===================== Question Id = ",question_list[i]['question_id']," ======================")
    # message = "What is the capital of France?"
    message = question_list[i]['turns'][0]
    # message = "Hello"
    conv = get_conversation_template("llama-2-chat")  
    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p
    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " "

    input_ids=tokenizer([prompt]).input_ids
    seqlen = len(input_ids[0])
    input_ids = torch.as_tensor(input_ids).cuda()
    output_ids = model.generate(input_ids, max_length=seqlen+256,temperature = 0, do_sample = False)
    output_ids_tot += len(output_ids[0]) - seqlen
    output=tokenizer.decode(output_ids[0])
    # print(output)
ed = time.time()

print('Time:',ed -st)
print('average time ',(ed-st)/output_ids_tot)
