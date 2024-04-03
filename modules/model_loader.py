import io
import os
import math
import tqdm
import torch
import numpy as np
import torch.nn as nn
from time import time
from torch.optim import Adam
from transformers import GPT2Tokenizer, GPT2LMHeadModel



class SVDConv1D(nn.Module):
    def __init__(self, original_layer, rank=None):
        super(SVDConv1D, self).__init__()
        self.nf=original_layer.nf
        U, S, V = torch.svd(original_layer.weight.data)
        if rank is not None and rank < len(S):
            U = U[:, :rank]
            S = S[:rank]
            V = V[:, :rank]
        U = nn.Parameter(U,requires_grad=False)
        S = nn.Parameter(torch.diag(S),requires_grad=False)
        self.V = nn.Parameter(V)
        self.precomputed_SU = nn.Parameter(U @ S,requires_grad=False)
        self.bias = nn.Parameter(original_layer.bias.data)
    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = x.view(-1, x.size(-1))
        x = x @ self.precomputed_SU
        x = x @ self.V.t()
        # x = x * self.alpha
        x = x + self.bias # *self.beta
        x = x.view(size_out)
        return x

class LoraConv1D(nn.Module):
    def __init__(self, c_attn_layer, rank=10):
        super().__init__()
        self.c_attn = c_attn_layer
        self.rank = rank

        self.A = nn.Parameter(torch.randn(self.c_attn.weight.shape[0], rank))
        self.B = nn.Parameter(torch.randn(rank, self.c_attn.weight.shape[1]))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.c_attn.nf,)
        delta_W = self.A @ self.B
        adapted_weight = self.c_attn.weight + delta_W
        x = torch.addmm(self.c_attn.bias, x.view(-1, x.size(-1)), adapted_weight)
        x = x.view(size_out)
        return x

def apply_Lowrank(model,layers,ranks,layer_names,device='cuda',lowrank_method='lora'):
    assert(len(layers)==len(ranks))
    assert(len(layer_names)==len(ranks))
    assert(lowrank_method in ['lora','svd'])
    np_old=count_param(model)
    for ind in range(len(layers)):
        if layer_names[ind]=='mlp_c_fc':
            if lowrank_method=='lora':
                model.transformer.h[layers[ind]].mlp.c_fc = LoraConv1D(model.transformer.h[layers[ind]].mlp.c_fc,rank=ranks[ind])
            if lowrank_method=='svd':
                model.transformer.h[layers[ind]].mlp.c_fc = SVDConv1D(model.transformer.h[layers[ind]].mlp.c_fc,rank=ranks[ind])

        if layer_names[ind]=='mlp_c_proj':
            if lowrank_method=='lora':
                model.transformer.h[layers[ind]].mlp.c_proj = LoraConv1D(model.transformer.h[layers[ind]].mlp.c_proj,rank=ranks[ind])
            if lowrank_method=='svd':
                model.transformer.h[layers[ind]].mlp.c_proj = SVDConv1D(model.transformer.h[layers[ind]].mlp.c_proj,rank=ranks[ind])


        if layer_names[ind]=='attn_c_attn':
            if lowrank_method=='lora':
                model.transformer.h[layers[ind]].attn.c_attn = LoraConv1D(model.transformer.h[layers[ind]].attn.c_attn,rank=ranks[ind])
            if lowrank_method=='svd':
                model.transformer.h[layers[ind]].attn.c_attn = SVDConv1D(model.transformer.h[layers[ind]].attn.c_attn,rank=ranks[ind])

        if layer_names[ind]=='attn_c_proj':
            if lowrank_method=='lora':
                model.transformer.h[layers[ind]].attn.c_proj = LoraConv1D(model.transformer.h[layers[ind]].attn.c_proj,rank=ranks[ind])
            if lowrank_method=='svd':
                model.transformer.h[layers[ind]].attn.c_proj = SVDConv1D(model.transformer.h[layers[ind]].attn.c_proj,rank=ranks[ind])


    model.to(device)
    np_new=count_param(model)
    reduction_percent= round((np_old-np_new)/np_old,2)*100
    if reduction_percent>0:
        print(f"The number of trainable params in your model reduced from {np_old} to {np_new} ({reduction_percent}%).")
    else:
        print(f"The number of trainable params in your model increase from {np_old} to {np_new} ({-reduction_percent}%).")
    return model


def count_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def load_model_utils(lr=1e-5,device='cuda'):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    return model,tokenizer, optimizer, loss_function

def evaluate_perplexity(model, test_loader, tokenizer):
    model.eval()
    total_loss = 0
    total_words = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_words += inputs["input_ids"].size(1)

    average_loss = total_loss / total_words
    perplexity = math.exp(average_loss)
    return perplexity
def train(model,train_loader,test_loader,
          tokenizer,optimizer,loss_function, 
          epochs=5):
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        time_records=[]
        for batch in tqdm.tqdm(train_loader):
            start_time=time()
            outputs=model(**batch,labels=batch['input_ids'])
            loss=outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            time_records.append(time() - start_time)
        average_loss=total_loss/len(train_loader)
        print(f"EPOCH: {epoch}, loss: {average_loss}, test_perplexity: {evaluate_perplexity(model, test_loader, tokenizer)}")
    return np.mean(time_records)





def evaluate_inference_time(model,test_loader,tokenizer,device,iters=32):
    model.to(device)
    with torch.no_grad():
        time_records=[]
        batch=next(iter(test_loader))
        for _ in range(iters):
            for i in range(batch['input_ids'].shape[0]):
                inputs = {key:batch[key][i:i+1].to(device) for key in batch.keys()}
                start_time=time()
                outputs = model(**inputs)
                time_records.append(time() - start_time)
    return np.mean(time_records)

def count_all_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())

def get_model_size(model):
    with torch.no_grad():
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        model_size_bytes = buffer.getbuffer().nbytes
    return model_size_bytes



def run_pipeline(model,train_loader,test_loader,
          tokenizer,optimizer,loss_function, 
          device,epochs=5,name=None):
    print('model_size calculation...')
    model_size=get_model_size(model)
    print('params counting...')
    trainable_params,all_params=count_all_param(model)
    print('inference time analysis...')
    cpu_inference_time=evaluate_inference_time(model,test_loader,tokenizer,'cpu')
    gpu_inference_time=evaluate_inference_time(model,test_loader,tokenizer,'cuda')
    print('evaluation before training...')
    base_perplexity=evaluate_perplexity(model,test_loader,tokenizer)
    print('training...')
    training_time=train(model,train_loader,test_loader,
          tokenizer,optimizer,loss_function, 
          epochs)
    print('evaluation after training...')
    after_train_perplexity=evaluate_perplexity(model,test_loader,tokenizer)
    print("FINISHED.")
    return {'name':name,'model_size':model_size,'trainable_params':trainable_params,'all_params':all_params,\
        'cpu_inference_time':cpu_inference_time,'gpu_inference_time':gpu_inference_time, \
        'base_perplexity':base_perplexity, 'training_time':training_time, 'after_train_perplexity':after_train_perplexity}

