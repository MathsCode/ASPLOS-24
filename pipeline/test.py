from eagle.model.ea_model import EaModel
import torch
import torch.multiprocessing as mp


class test():
    def __init__(self) -> None:
        self.stream = torch.cuda.Stream()
    def work(self,model,hidden_states,input_ids_draft):
        with torch.cuda.stream(self.stream):
            for j in range(4):
                output2 = model(hidden_states,input_ids_draft)
        
    
def test2(model,input_ids):
    for i in range(10):
        model(input_ids)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    model = EaModel.from_pretrained(
        base_model_path='/share/datasets/public_models/Llama-2-7b-chat-hf',
        ea_model_path="/home/xujiaming/xujiaming/datasets/LLM_models/EAGLE-llama2-chat-7B",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        is_offload = False
    )
    model.eval()
    message = "What is the capital of France?"
    input_ids=model.tokenizer([message]).input_ids
    input_ids_verify = [[666 for i in range(26)]]
    input_ids_draft = [[666 for i in range(4)]]
    length = len(input_ids_draft[0])
    input_ids_draft = torch.as_tensor(input_ids_draft).cuda()
    input_ids_verify = torch.as_tensor(input_ids_verify).cuda()
    
    hidden_states = torch.ones((1,length,4096)).to(input_ids_draft.device).half()

    
    

    for i in range(128):
        output1 = model.base_model(input_ids_verify)
        output1 = model.base_model(input_ids_verify)
        for j in range(4):
            output2 = model.ea_layer(hidden_states,input_ids_draft)
        torch.cuda.synchronize()
        
        
    import time
    st = time.time()

    for i in range(256):
        output1 = model.base_model(input_ids_verify)
        for j in range(4):
            output2 = model.ea_layer(hidden_states,input_ids_draft)
        output1 = model.base_model(input_ids_verify)
        torch.cuda.synchronize()
    ed = time.time()
    print("Sequential Time",ed - st)

    stream1 = torch.cuda.Stream()
    import time
    st = time.time()
    t = test()
    for i in range(256):
        output1 = model.base_model(input_ids_verify)
        with torch.cuda.stream(stream1):
            for j in range(4):
                output2 = model.ea_layer(hidden_states,input_ids_draft)
        output1 = model.base_model(input_ids_verify)   
        # for j in range(4):
        #     output2 = model.ea_layer(hidden_states,input_ids_draft)
        torch.cuda.synchronize()
    ed = time.time()
    print(output1)
    print(output2)
    # print(tot1)
    print("Parallel Time",ed - st)
        
    
    

