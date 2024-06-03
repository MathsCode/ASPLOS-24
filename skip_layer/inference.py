from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
model = EaModel.from_pretrained(
    base_model_path='/share/datasets/public_models/Llama-2-7b-chat-hf',
    ea_model_path="/home/xujiaming/xujiaming/datasets/LLM_models/EAGLE-llama2-chat-7B",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()
message = "hello"
conv = get_conversation_template("llama-2-chat")  
sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
conv.system_message = sys_p
conv.append_message(conv.roles[0], message)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt() + " "