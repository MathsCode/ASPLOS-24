import copy
import json
import time

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from .choices import mc_sim_7b_63
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download
from .cnets import Model
from .configs import EConfig
from huggingface_hub import hf_hub_download

import time



class EaModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            
            # [xjm:] offload
            is_offload = False,
            skip_model = None,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        self.ea_layer = Model(config,bias=bias)

        low_memory=False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device!=base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                # self.ea_layer.head=nn.Linear(base_model.lm_head.in_features,base_model.lm_head.out_features,bias=False)
                # self.ea_layer.head.weight=copy.deepcopy(base_model.lm_head.weight)
                # self.ea_layer.head.to(device)
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        
        if is_offload:
            # self.ea_layer.to(self.base_model.dtype)
            self.ea_layer.to(torch.bfloat16)
            print("draft model device",self.ea_layer.layers[0].self_attn.q_proj.weight.device)
            import copy
            self.ea_layer_lm_head = copy.deepcopy(self.base_model.lm_head).to(torch.bfloat16).to(self.ea_layer.layers[0].self_attn.q_proj.weight.device)
            self.ea_layer_lm_head.eval()
            self.ea_layer.init_tree()
        else:
            self.ea_layer.to(self.base_model.dtype).to(device)
            self.ea_layer.init_tree()
            self.ea_layer_lm_head = self.base_model.lm_head
            
        if skip_model is not None:
            import lightgbm as lgb
            self.skip_model = lgb.Booster(model_file=skip_model)
        else:
            self.skip_model = None

            
        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            is_offload = False,
            skip_model = None,
            **kwargs,
    ):
        #assert Type=="LLaMA" or "Mixtral"
        Type=AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type=='LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath=os.path.join(ea_model_path,"config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")
        model = cls(
            base_model,
            base_model_path,
            configpath,
            is_offload = is_offload,
            skip_model = skip_model,
        )
        load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
        if not os.path.exists(load_model_path):
            load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
        ea_layer_state_dict = torch.load(load_model_path,
                                         map_location=base_model.device)
        model.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            init=True,
            logits_processor=None,
            
            # [xjm:] Add param for all hidden states
            output_hidden_states = None,
            # [xjm:] add skip layer param
            skip_layer = None,
            # [xjm:] add exit_layer param
            exit_layer = None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            st = time.time()
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                
                # [xjm:] add output hidden states
                output_hidden_states = output_hidden_states,
                # [xjm:] add skip layer param
                skip_layer = skip_layer,
                # [xjm:] add exit_layer param
                exit_layer = exit_layer,
            )
            Tv = time.time() - st
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0].clone()
        if init:
            if logits_processor is not None:
                logits = orig[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                token = torch.multinomial(probabilities, 1)
            else:
                token = torch.argmax(orig[:, -1])
                token = token[None, None]
            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            # Clone the output hidden states
            st = time.time()
            # ea_logits = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head, logits_processor)
            ea_logits = self.ea_layer.topK_genrate(hidden_states, input_ids, self.ea_layer_lm_head, logits_processor)
            Td = time.time() - st
            
            # [xjm:] add return time Tv, Td
            if output_orig:
                return ea_logits, outputs, orig, hidden_states, token, Tv,Td
            return ea_logits, hidden_states, token
        else:
            if output_orig:
                # [xjm:] add return time Tv
                return outputs, orig, hidden_states,Tv

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            tree_choices=mc_sim_7b_63,
            skip_layer = False,
            Early_Exiting = False,
            time_breakdown = False,
            exe_layers = [0,0,0,0,0],
            exe_tokens = [0,0,0,0,0],
            max_value_dist = [[]],
            train_data = [],
            train_label = [],
            train_hidden_states = [],
    ):
        all_st = time.time()
        assert (skip_layer and Early_Exiting) == False,"Can not support both skip layer and Early Exiting"
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        tree_logits, logits, hidden_state, sample_token,Tv,Td = initialize_tree(
            input_ids, self, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
        )
        
        new_token = 0
        tot_skip_layers = []
        tot_exit_layers = []
        draft_time = Td
        verify_time = Tv
         
        for idx in range(max_length):
            
            candidates, cart_candidates_prob, tree_candidates,exp_candidates_prob = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                logits_processor
            )
            
            
            skipped_layers = None
            
            if self.skip_model is not None:
                predict_X = torch.cat((torch.broadcast_to(tree_candidates,(self.config.num_hidden_layers,tree_candidates.shape[-1])),torch.arange(0,self.config.num_hidden_layers).reshape(-1,1).to(tree_candidates.device)),dim = 1)
                predict_X = predict_X.cpu().numpy()
                result = self.skip_model.predict(predict_X)
                tmp = []
                for i in range(self.config.num_hidden_layers):
                    if result[i] > 0.6:
                        tmp.append(i)
                skipped_layers = tmp
                # print(skipped_layers)
            # [xjm:] add copy for skip layer
            if(skip_layer or Early_Exiting):
                import copy
                key_values_back =  copy.deepcopy(past_key_values)
                
            
            logits, hidden_state_new, outputs,Tv = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices_head"],
                # [xjm:] add skip layer
                skip_layer= skipped_layers,
                # [xjm:] add branch prediction
                # Early_Exiting = result[idx]
            )
            verify_time += Tv
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
                tree_candidates, tree_buffers["b_indices"]
            )
            
            # print(exp_candidates_prob[0])
            # print(exp_candidates_prob[1])
            # acc = [4,3,2,2,1,3,2,2,1,3,2]
            # cnt = 1
            # for ii in range(len(acc)):
            #     num = 0
            #     tot = 0
            #     while(num < acc[ii]):
            #         tot += exp_candidates_prob[1][cnt]
            #         num += 1
            #         cnt += 1
            #     print(ii," : ",tot)
            # print(best_candidate,accept_length)
                
                
              
            if(skip_layer):
                # [xjm:] check skip layer
                prob = sample_p
                if logits_processor is not None:
                    token = torch.multinomial(prob, 1)
                    token = token[None]
                else:
                    token = torch.argmax(prob)
                    token = token[None, None]
                # print('==============Best================')
                # print(best_candidate)
                # print(accept_length)
                # print(token)
                def skip_layer_func(idx_list):
                    cur_layers = 0
                    tmp_past_key_values = copy.deepcopy(key_values_back)
                    tmp_logits, tmp_hidden_state_new, tmp_outputs,Tv = tree_decoding(
                        self,
                        tree_candidates,
                        tmp_past_key_values,
                        tree_buffers["tree_position_ids"],
                        input_ids,
                        tree_buffers["retrieve_indices_head"],
                        # [xjm:] add skip layer
                        skip_layer=idx_list
                    )
                    del tmp_past_key_values
                    
                    tmp_candidate, tmp_accept_length, tmp_sample_p = evaluate_posterior(
                    tmp_logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
                    tree_candidates, tree_buffers["b_indices"]
                    )
                    
                    tmp_prob = tmp_sample_p
                    if logits_processor is not None:
                        tmp_token = torch.multinomial(tmp_prob, 1)
                        tmp_token = tmp_token[None]
                    else:
                        tmp_token = torch.argmax(tmp_prob)
                        tmp_token = tmp_token[None, None]
                    if(best_candidate == tmp_candidate and accept_length == tmp_accept_length and token == tmp_token):
                        return True
                    return False
                def search(idx_list,max_result,max_skip_layers,start_id):
                    # if(max_skip_layers > self.config.num_hidden_layers * 0.7):
                    #     return max_result,max_skip_layers
                    back_list = copy.deepcopy(idx_list)
                    for i in range(start_id,self.config.num_hidden_layers):
                        idx_list.append(i)
                        if len(idx_list) > max_skip_layers:
                            if skip_layer_func(idx_list):
                                # print('==========')
                                # print(idx_list)
                                max_skip_layers = len(idx_list)
                                max_result = idx_list
                            # elif len(idx_list) == max_skip_layers:
                            #     max_result.append(idx_list)
                                max_result,max_skip_layers = search(idx_list,max_result,max_skip_layers,start_id=i+1)
                        del idx_list
                        idx_list = copy.deepcopy(back_list)
                                
                    return max_result,max_skip_layers
                idx_list = []
                max_result,max_skip_layers = search(idx_list,[],0,0)
                print("==========Final==========")
                print(max_skip_layers)
                print(max_result)
                for id in range(self.config.num_hidden_layers):
                    train_data = torch.concat((tree_candidates,torch.tensor(id).to(tree_candidates.device).reshape(1,1)),dim=1)
                    if id in max_result:
                        train_label = torch.tensor(1).reshape(1)
                    else:
                        train_label = torch.tensor(0).reshape(1)
                    train_data_path = '/home/xujiaming/xujiaming/research/ASPLOS-24/results/train_data_vi7B.pt'
                    train_lable_path = '/home/xujiaming/xujiaming/research/ASPLOS-24/results/train_lable_vi7B.pt'
                    if os.path.exists(train_data_path):
                        data = torch.load(train_data_path)
                        train_data= torch.concat((data,train_data),dim=0)
                        torch.save(train_data,train_data_path)
                        data = torch.load(train_lable_path)
                        train_label = torch.concat((data,train_label))
                        torch.save(train_label,train_lable_path)
                    else:
                        torch.save(train_data,train_data_path)
                        torch.save(train_label,train_lable_path)
        
            '''
            if(skip_layer):
                skip_layer_list = []
                # [xjm:] check skip_layer
                prob = sample_p
                if logits_processor is not None:
                    token = torch.multinomial(prob, 1)
                    token = token[None]
                else:
                    token = torch.argmax(prob)
                    token = token[None, None]
                print('==============Best================')
                print(best_candidate)
                print(accept_length)
                print(token)
                
                for id in range(self.config.num_hidden_layers):
                    tmp_past_key_values = copy.deepcopy(key_values_back)
                    tmp_logits, tmp_hidden_state_new, tmp_outputs,Tv = tree_decoding(
                        self,
                        tree_candidates,
                        tmp_past_key_values,
                        tree_buffers["tree_position_ids"],
                        input_ids,
                        tree_buffers["retrieve_indices_head"],
                        # [xjm:] add skip layer
                        skip_layer=id
                    )
                    del tmp_past_key_values
                    
                    tmp_candidate, tmp_accept_length, tmp_sample_p = evaluate_posterior(
                    tmp_logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
                    tree_candidates, tree_buffers["b_indices"]
                    )
                    tmp_prob = tmp_sample_p
                    if logits_processor is not None:
                        tmp_token = torch.multinomial(tmp_prob, 1)
                        tmp_token = tmp_token[None]
                    else:
                        tmp_token = torch.argmax(tmp_prob)
                        tmp_token = tmp_token[None, None]
                    train_data = torch.concat((tree_candidates,torch.tensor(id).to(tree_candidates.device).reshape(1,1)),dim=1)
                    if best_candidate == tmp_candidate and accept_length == tmp_accept_length and token == tmp_token:
                        train_label = torch.tensor(1).reshape(1)
                    else:
                        train_label = torch.tensor(0).reshape(1)
                    train_data_path = '/home/xujiaming/xujiaming/research/ASPLOS-24/results/train_data.pt'
                    train_lable_path = '/home/xujiaming/xujiaming/research/ASPLOS-24/results/train_lable.pt'
                    if os.path.exists(train_data_path):
                        data = torch.load(train_data_path)
                        train_data= torch.concat((data,train_data),dim=0)
                        torch.save(train_data,train_data_path)
                        data = torch.load(train_lable_path)
                        train_label = torch.concat((data,train_label))
                        torch.save(train_label,train_lable_path)
                    else:
                        torch.save(train_data,train_data_path)
                        torch.save(train_label,train_lable_path)
            '''
            if(Early_Exiting):
                # [xjm:] check Early Exiting
                prob = sample_p
                if logits_processor is not None:
                    token = torch.multinomial(prob, 1)
                    token = token[None]
                else:
                    token = torch.argmax(prob)
                    token = token[None, None]
                # print('==============Accurate================')
                # print(candidates[best_candidate])
                # print(accept_length)
                # print(token)
                
                cur_layers = 0
                flag = [True] * 5
                for id in range(self.config.num_hidden_layers):
                    tmp_past_key_values = copy.deepcopy(key_values_back)
                    tmp_logits, tmp_hidden_state_new, tmp_outputs,Tv = tree_decoding(
                        self,
                        tree_candidates,
                        tmp_past_key_values,
                        tree_buffers["tree_position_ids"],
                        input_ids,
                        tree_buffers["retrieve_indices_head"],
                        # [xjm:] add exit layer,
                        exit_layer=id
                    )
                    del tmp_past_key_values
                    
                    tmp_candidate, tmp_accept_length, tmp_sample_p,cur_result = evaluate_posterior(
                    tmp_logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
                    tree_candidates, tree_buffers["b_indices"],early_exiting=True
                    )
                    tmp_prob = tmp_sample_p
                    if logits_processor is not None:
                        tmp_token = torch.multinomial(tmp_prob, 1)
                        tmp_token = tmp_token[None]
                    else:
                        tmp_token = torch.argmax(tmp_prob)
                        tmp_token = tmp_token[None, None]
                    # if best_candidate == tmp_candidate and accept_length == tmp_accept_length and token == tmp_token:
                    #     print('[YES]==============Exit layer id = ', id,'================')
                    #     print(best_candidate,' v.s.',tmp_candidate)
                    #     print(accept_length, 'v.s.',tmp_accept_length)
                    #     print(token, 'v.s.',tmp_token)
                    #     break
                    prob_soft = torch.nn.functional.softmax(tmp_logits,dim=-1)
                    prob_soft_cand = torch.nn.functional.softmax(tmp_logits[0,0,tree_candidates[0,1:5]],dim=-1)
                    max_value_dist[id].append(float(torch.max(tmp_logits[0,0])))
                    
                    train_hidden_states.append(tmp_hidden_state_new[0,0].unsqueeze(0))
                    train_data.append(tmp_logits[0,0,tree_candidates[0,1:5]].unsqueeze(0))
                    
                    train_label.append(cur_result[best_candidate][0].reshape(-1))
                    # print(id,' : ',cur_result[best_candidate])
                    # print(id,' : ',tmp_logits[0,0,tree_candidates[0,1:5]])
                    # print(id,' : ',prob_soft[0,0,tree_candidates[0,1:5]])
                    # print(id,' : ',prob_soft_cand)
                    # print(id,' : ',torch.max(tmp_logits[0,0]))
                    
                    for i in range(accept_length):
                        if flag[i]:
                            if cur_result[best_candidate][i] == 1:
                                exe_layers[i] +=  id + 1
                                exe_tokens[i] += 1
                                flag[i] = False
                    # else:
                    #     print('[No ]==============Exit layer id = ', id,'================')
            
    
            
            input_ids, tree_logits, new_token, hidden_state, sample_token,Td = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits_processor,
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state,
                hidden_state_new,
                sample_p
            )
            draft_time += Td
            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                all_time = time.time() - all_st
                if time_breakdown:
                    print("All Time:",all_time)
                    print("Draft Time:", draft_time, draft_time / all_time)
                    print("Verify Time:", verify_time, verify_time / all_time)
                return input_ids
            if new_token > max_new_tokens:
                all_time = time.time() - all_st
                if time_breakdown:
                    print("All Time:",all_time)
                    print("Draft Time:", draft_time, draft_time / all_time)
                    print("Verify Time:", verify_time, verify_time / all_time)
                return input_ids
            if input_ids.shape[1] > max_length:
                all_time = time.time() - all_st
                if time_breakdown:
                    print("All Time:",all_time)
                    print("Draft Time:", draft_time, draft_time / all_time)
                    print("Verify Time:", verify_time, verify_time / all_time)
                return input_ids
        
        all_time = time.time() - all_st
        if time_breakdown:
            print("All Time:",all_time)
            print("Draft Time:", draft_time, draft_time / all_time)
            print("Verify Time:", verify_time, verify_time / all_time)
    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_steps=512,
            tree_choices=mc_sim_7b_63,

    ):
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        tree_logits, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
        )
        new_token = 0

        for idx in range(max_steps):
            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                logits_processor
            )
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices_head"],
            )

            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
                tree_candidates, tree_buffers["b_indices"]
            )
            #print("post", time.time() - s)
            input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits_processor,
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_steps=512,
            tree_choices=mc_sim_7b_63,

    ):
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0

        for idx in range(max_steps):
            input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)

            yield input_ids

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break
