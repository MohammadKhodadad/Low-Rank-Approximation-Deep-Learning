import math
import tqdm
import torch
from torch.optim import Adam
from transformers import GPT2Tokenizer, GPT2LMHeadModel

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
        for batch in tqdm.tqdm(train_loader):
            outputs=model(**batch,labels=batch['input_ids'])
            loss=outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss=total_loss/len(train_loader)
        print(f"EPOCH: {epoch}, loss: {average_loss}, test_perplexity: {evaluate_perplexity(model, test_loader, tokenizer)}")