import torch

logit = torch.tensor([[1,2,3,4],[5,6,7,8]])
target = torch.tensor([[1,2,3,4],[5,6,7,8]])


logit = logit[..., :-1, :].contiguous()
 #   .view(-1, logit.size(-1))
labels = target[..., 1:].contiguous()
 #   .view(-1)

print(logit)
print(labels)