import torch
train_data = torch.load('/home/xujiaming/xujiaming/research/ASPLOS-24/results/train_data.pt')
train_label = torch.load('/home/xujiaming/xujiaming/research/ASPLOS-24/results/train_label.pt')
# train_label = train_label.cpu().numpy()
# train_data_soft = torch.nn.functional.softmax(train_data,dim=-1)

import numpy as np
np.set_printoptions(precision=4, suppress=True)
for i in range(0,train_data.shape[0],32):
    result_descend = train_data[i].sort(descending=True).values
    result_descend_soft = torch.nn.functional.softmax(result_descend,dim=-1)
    top_logits = result_descend[0]
    second_logits = result_descend[1]
    top_prob = result_descend_soft[0]
    second_prob = result_descend_soft[1]
    if train_label[i] == 0:
        # print(i,train_data[i],train_data_soft[i],train_label[i].item())
        print(float(top_prob/second_prob),float(top_logits-second_logits),float(top_logits/second_logits),float(top_logits),float(top_prob))
        

# feature = torch.cat([train_data,train_data_soft],dim=-1)
# torch.save(feature,'/home/xujiaming/xujiaming/research/ASPLOS-24/results/feature.pt')
