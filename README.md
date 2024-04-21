# Efficient LLMs via Low-Rank Approximation
In this project we are going to design modules that can help us facilitate the process of low rank approximation.
This project will mostly be educational and have a few very useful modules that you can integrate in you code.

What we will cover is LoRA and SVD reduction on LMs.

## How to reproduce the results
### Dependencies 
This project involves training neural networks and large language models. To install the dependencies, you have two options: GPU or CPU (it can take days to train GPT-2 though). 
First, you have to clone this repository (if you have received a zip file containing this repo, simply ignore this). 
For CPU, simply run:
```
pip install torch
pip install -r requirements.txt
```
For GPU, run:
```
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
Note that you need to have `python` and `pip` installed on your system/server. For training with GPU, CUDA should be installed on your machine. To check the version, simply run `nvcc --version`. The above code for GPU installs `pytorch` for CUDA version 11.8. If you have any other versions of CUDA, refer to [here](https://pytorch.org/get-started/previous-versions/) and install the latest `pytorch` version that supports your CUDA installation. It is the best practice to create a virtual environment then install packages in it. 

To run a sample example of Truncated-SVD adaptation for a small neural network and train it on the MNIST dataset, you can refer to the `experiments/svd_test.ipynb` notebook. you can open python notebooks with VSCode, if you get any messages regarding installing dependencies to run a notebook, press install.

To fully reproduce the results in the report, you can run the following code:
```
python experiments/train_and_exps.py
```
Depending on your GPU, it can takes hours or maybe an entire day or more to train the models (it took us around 2 days) and output the results. Also, if you are a Google Colab Pro or Colab Pro+ subscriber, you can use [this link](https://colab.research.google.com/drive/1PCMhLab-X0ypIu4ys9ZR4B3fCUe-9WIW?copy#scrollTo=dGMYD7xp9RH7) to reproduce the results. We have reduced the batch size to 4 (we used 128) in order for training to be more tractable, but still we estimate the training of GPT-2 base to take 36 hours on Google Colab's T4 GPU. 