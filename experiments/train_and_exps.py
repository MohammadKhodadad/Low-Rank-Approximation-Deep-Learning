


import json
import glob
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

import sys
from time import time
sys.path.append('..')
from modules.data_loader import *
from modules.model_loader import *


n_epochs=128
print(n_epochs)
device = "cuda" if torch.cuda.is_available() else "cpu"
model,tokenizer,optimizer,loss_function = load_model_utils(device=device)
train_dataset,test_dataset,train_loader,test_loader=pipeline_load_dataset(tokenizer,device=device,batch_size=32)
gpt_results=run_pipeline(model,train_loader,test_loader,
          tokenizer,optimizer,loss_function, 
          device,n_epochs,'gpt2')



device = "cuda" if torch.cuda.is_available() else "cpu"
model,tokenizer,optimizer,loss_function = load_model_utils(device=device)
train_dataset,test_dataset,train_loader,test_loader=pipeline_load_dataset(tokenizer,device=device,batch_size=32)


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
          device,n_epochs,'svd')



device = "cuda" if torch.cuda.is_available() else "cpu"
model,tokenizer,optimizer,loss_function = load_model_utils(device=device)
train_dataset,test_dataset,train_loader,test_loader=pipeline_load_dataset(tokenizer,device=device,batch_size=32)


layers=[]
ranks=[]
layer_names=[]
for layer in range(12):
    for layer_name in ['mlp_c_fc','mlp_c_proj','attn_c_attn','attn_c_proj']:
        layers.append(layer)
        ranks.append(32)
        layer_names.append(layer_name)

for param in model.parameters():
        param.requires_grad = False

model=apply_Lowrank(model,layers=layers,ranks=ranks,layer_names=layer_names,device=device,lowrank_method='lora')


svd_results=run_pipeline(model,train_loader,test_loader,
          tokenizer,optimizer,loss_function, 
          device,n_epochs,'lora')



def visualize_single(address):
    with open(address,'r') as f:
        results=json.load(f)
    history=pd.DataFrame.from_dict(results['history'])

    # fig_loss = px.line(history, x='epoch', y='train_loss', title=f'Loss vs Time for {results["name"]}',
    #                 labels={'Loss': 'Loss', 'Time': 'Time (seconds)'},
    #                 markers=True)
    # fig_loss.update_layout(
    #     title_font_size=20,
    #     title_x=0.5,
    #     # plot_bgcolor='white',
    #     xaxis=dict(title='Time (s)'),
    #     yaxis=dict(title='Loss'),
    # )
    # fig_loss.update_traces(marker=dict(size=5))
    # fig_loss.show()
    # fig_loss.write_image(f'loss_{results["name"]}.pdf')

    # # Plot for Perplexity vs Time
    # fig_perplexity = px.line(history, x='epoch', y='test_perplexity', title=f'Perplexity vs Time for {results["name"]}',
    #                         labels={'Perplexity': 'Perplexity', 'Time': 'Time (seconds)'},
    #                         markers=True)
    # fig_perplexity.update_layout(
    #     title_font_size=20,
    #     title_x=0.5,
    #     # plot_bgcolor='white',
    #     xaxis=dict(title='Time (s)'),
    #     yaxis=dict(title='Perplexity', type='log'),
    # )
    # fig_loss.update_traces(marker=dict(size=5))
    # fig_perplexity.show()
    # fig_loss.write_image(f'perplexity_{results["name"]}.pdf')
    

    # Assuming 'history' is a DataFrame with columns 'epoch', 'train_loss', and 'test_perplexity'

    # Plot for Loss vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_loss'], marker='o', label='Train Loss')
    plt.title(f'Loss vs Time for {results["name"]}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'loss_{results["name"]}.pdf')
    plt.show()

    # Plot for Perplexity vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['test_perplexity'], marker='o', label='Test Perplexity')
    plt.title(f'Perplexity vs Time for {results["name"]}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Perplexity')
    plt.yscale('log')  # Setting y-axis to logarithmic scale for perplexity
    plt.grid(True)
    plt.savefig(f'perplexity_{results["name"]}.pdf')
    plt.show()

    return pd.DataFrame({key:[results[key]] for key in results.keys() if key !='history'})




def visualize_bulk(addresses):
    results=[]
    for address in addresses:
        results.append(visualize_single(address))
    return pd.concat(results,axis=0)




addresses=glob.glob('../results/*.json')
visualize_bulk(addresses)