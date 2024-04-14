import sys
from time import time
sys.path.append('..')
from modules.data_loader import *
from modules.model_loader import *



device = "cuda" if torch.cuda.is_available() else "cpu"
model,tokenizer,optimizer,loss_function = load_model_utils(device=device)
train_dataset,test_dataset,train_loader,test_loader=pipeline_load_dataset(tokenizer,device=device)
gpt_results=run_pipeline(model,train_loader,test_loader,
          tokenizer,optimizer,loss_function, 
          device,30,'gpt2')



device = "cuda" if torch.cuda.is_available() else "cpu"
model,tokenizer,optimizer,loss_function = load_model_utils(device=device)
train_dataset,test_dataset,train_loader,test_loader=pipeline_load_dataset(tokenizer,device=device)


layers=[]
ranks=[]
layer_names=[]
for layer in range(12):
    for layer_name in ['mlp_c_fc','mlp_c_proj','attn_c_attn','attn_c_proj']:
        layers.append(layer)
        ranks.append(32)
        layer_names.append(layer_name)
model=apply_Lowrank(model,layers=layers,ranks=ranks,layer_names=layer_names,device=device,lowrank_method='svd')


svd_results=run_pipeline(model,train_loader,test_loader,
          tokenizer,optimizer,loss_function, 
          device,30,'svd')



device = "cuda" if torch.cuda.is_available() else "cpu"
model,tokenizer,optimizer,loss_function = load_model_utils(device=device)
train_dataset,test_dataset,train_loader,test_loader=pipeline_load_dataset(tokenizer,device=device)


layers=[]
ranks=[]
layer_names=[]
for layer in range(12):
    for layer_name in ['mlp_c_fc','mlp_c_proj','attn_c_attn','attn_c_proj']:
        layers.append(layer)
        ranks.append(32)
        layer_names.append(layer_name)
model=apply_Lowrank(model,layers=layers,ranks=ranks,layer_names=layer_names,device=device,lowrank_method='lora')


svd_results=run_pipeline(model,train_loader,test_loader,
          tokenizer,optimizer,loss_function, 
          device,30,'lora')