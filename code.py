#!/usr/bin/env python
# coding: utf-8

# # Importing Modules

# In[5]:


import nltk
from tqdm import tqdm
import math
from operator import add
nltk.download('stopwords')
nltk.download('punkt')
import gc
import numpy as np
from scipy import spatial
from nltk import tokenize
from sklearn.model_selection import train_test_split
import random
import os
import gensim.models.keyedvectors as word2vec
import pickle
import torch
import torch.nn as nn
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
import threading
import operator
from nltk.corpus import stopwords


# In[6]:


MAX_WORDS = 18
MAX_SENTENCE = 12
ENC_VECTOR = 300


# In[7]:


# Hyperparameters
input_size = MAX_WORDS * ENC_VECTOR
sequence_length = MAX_SENTENCE
num_layers = 1
hidden_size = 256
num_classes = 2
learning_rate = 0.03
batch_size = 40
num_epochs = 300


# # Loading Dataset
# 

# In[8]:


import pandas as pd
df =pd.read_csv("ISOT_train.csv")
df = df.dropna()
df['articleBody'] = df['Headline']+df['articleBody']
df = df.drop(['Body ID', 'Headline'], axis=1)
print(df.head())


# # PREPROCESSING

# In[9]:


stop = stopwords.words('english')
df['articleBody'] = df['articleBody'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
spec_chars_null = ["!","’",'"',"#","”","“","'","\"",",","$","%","&","'","(",")","*","+",",","-","/",":",";","<","=",">","?","@","[","\\","]","^","_","`","{","|","}","~","–"]
for char in spec_chars_null:
    df['articleBody'] = df['articleBody'].str.replace(char, '')
df['articleBody']


# In[10]:


#Data Preprocessing remaining 
# vocab : set of all the words in all the documents of corpus
def create_vocab(document_set):
  vocabulary = []
  for doc in document_set:
    for sentence in doc:
      vocabulary.append(sentence.split())
  return vocabulary

vocab = create_vocab(df[["articleBody"]].values.tolist())
flat_list = [item for sublist in vocab for item in sublist]
vocab = list(set(flat_list))

del flat_list


# In[11]:


word_vec_model = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


# In[12]:


dct={}
for word in vocab:
  if word in word_vec_model:
    dct[word] = word_vec_model[word]
  else:
    dct[word] = np.random.rand(1, ENC_VECTOR)[0]                                     ## EDIT
if not os.path.isdir("EmbedISOT"):
  os.mkdir("EmbedISOT")
pkl = open('EmbedISOT/word_embedding.pkl', 'wb')   
# source, destination
pickle.dump(dct, pkl)                     
pkl.close()

del dct
del word_vec_model


# In[13]:


obj = pd.read_pickle(r'EmbedISOT/word_embedding.pkl')


# In[14]:


def parse_document(doc):
  return tokenize.sent_tokenize(doc)


# In[15]:


def get_sentence_embedding(sent_list):
  doc_emb_list = []
  sent_count = 0
  for sent in sent_list:
    if sent_count >= MAX_SENTENCE:
        break
    sent_emb_list =[]
    count = 0
    for w in sent.split(" "):
      if count >= MAX_WORDS:
        break
      try:
        if w != '':
          sent_emb_list.append(obj[w])
          count += 1
      except:
        print('Error word : ', w)
        
    for _ in range(MAX_WORDS - count):
      temp = [0]*ENC_VECTOR
      sent_emb_list.append(np.array(temp))
    sent_emb_list = [item for sublist in sent_emb_list for item in sublist]
    
    doc_emb_list.append(sent_emb_list)
    sent_count+=1
    
  for _ in range(MAX_SENTENCE - len(sent_list)):
    temp = [0]*(ENC_VECTOR * MAX_WORDS)
    doc_emb_list.append(temp)

  del sent_emb_list

  return doc_emb_list


# In[16]:


def get_all_sent_in_doc(doc):
  all_sent = []
  for i in range(len(doc)):
    sent_list = parse_document(doc.loc[i, 'articleBody'])
    all_sent.append(sent_list)
  
  return all_sent


# In[17]:


def get_all_sent_emb_in_doc(doc, id):
  all_sent_emb = []
  all_doc_sent = []
  for i in range(len(doc)):
    sent_list = parse_document(doc.loc[i+id, 'articleBody'])
    xz = get_sentence_embedding(sent_list)
    all_sent_emb.append(xz)
    all_doc_sent.append(sent_list)
  
  return all_sent_emb, all_doc_sent


# # MODEL

# In[18]:


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[19]:


# Input : weight matrix { Shape : N x N }
# Output : index of core sentence
def core_sent_index(weight_matrix):
  max_index = 0
  max = 0
  for i in range(MAX_SENTENCE):
    for j in range(MAX_SENTENCE):
      weight_matrix[i][j] = weight_matrix[i][j].item()
  for i in range(MAX_SENTENCE):
    sum = 0
    for j in range(MAX_SENTENCE):
      if i == j:
        continue
      else:
        sum += weight_matrix[i][j]
    if sum > max:
      max = sum
      max_index = i
  return max_index


# In[20]:


# Input : sentence embedding {Shape : N x M} and index of core sentence
# Output : Relevant sentence embedding list and irrelevant sentence embedding list
def rel_irrel_sent(sentence_embedding, index_core_sent):
  sentence_embedding = sentence_embedding.cpu().detach().numpy()
  similarity = {}
  thres = MAX_SENTENCE * 0.3
  rel = []
  irrel = []

  for i in range(MAX_SENTENCE):
    similarity[i] = 1 - spatial.distance.cosine(sentence_embedding[i], sentence_embedding[index_core_sent])
  similarity = dict(sorted(similarity.items(), reverse = True, key =lambda kv:(kv[1], kv[0])))

  count = 1;
  for i in similarity:
    if count < thres:
      rel.append(i)
    else:
      irrel.append(i)
    count += 1
  return rel, irrel


# In[21]:


# Create a bidirectional LSTM
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, bidirectional=True)
        self.encoder_emb = nn.Sequential(nn.Linear(2*hidden_size, 50),nn.LeakyReLU(), nn.Dropout(0.3))
        self.w_fc = nn.Linear(100, 1)
        self.fc = nn.Linear(MAX_SENTENCE, num_classes)

    def forward(self, x):
        # Bi-LSTM Embedding
        # out : Corpus Document Embedding
        out, _ = self.lstm(x.float())

        single_document_embedding = []
        
        for temp_sen_emb in out:
          single_document_embedding.append(temp_sen_emb)
        
        doc_embedding_list = []
        corpus_iterator = 0
        # for corpus_iterator in range(len(x)):
        # Weight Matrix Calculation 
        for temp_sen_emb in single_document_embedding:
          enc_embedding = []
          for sen_emb in temp_sen_emb:
            enc_embedding.append(self.encoder_emb(sen_emb))    
          Wij = []
          for i in range(len(temp_sen_emb)):
            temp_row = []
            for j in range(len(temp_sen_emb)):
              if i==j:
                temp_row.append(torch.tensor(-9999999))
                continue
              else:
                temp_row.append(self.w_fc(torch.cat((enc_embedding[i], enc_embedding[j]))))
            Wij.append(temp_row)
          # Core Sentence Extraction
          core_sentence_index = core_sent_index(Wij)
          # print('Core Sentence Index : ', core_sentence_index)
          relevant_indexes, irrelevent_indexes = rel_irrel_sent(temp_sen_emb,core_sentence_index)
          
          #Generate Mapping of Vocabulary
          r = {}
          s = {}
          doc_vocab = []
          for sentence_iterator in all_doc_sent[corpus_iterator]:
            for word_iterator in sentence_iterator.split():
              doc_vocab.append(word_iterator)
          
          doc_vocab = list(set(doc_vocab))
          for word in doc_vocab:
            r[word] = 0
            s[word] = 0

          for i in relevant_indexes:
            if i >= len(all_doc_sent[corpus_iterator]):
              continue
            for word in all_doc_sent[corpus_iterator][i].split():
              r[word] +=1 

          for i in irrelevent_indexes:
            if i >= len(all_doc_sent[corpus_iterator]):
              continue
            for word in all_doc_sent[corpus_iterator][i].split():
              s[word] +=1 

          
          Rd = len(relevant_indexes)
          Sd = len(irrelevent_indexes)

          RS = {}

          for word in doc_vocab:
            RS[word] = math.log(abs(((r[word]+0.5)*(Sd-s[word]+0.5))/((Rd-r[word]-0.5)*(s[word]-0.5))))

          # doc should be special character removed
          sentence_score = {}
          
          for sentence_iterator,i in zip(all_doc_sent[corpus_iterator],range(len(all_doc_sent[corpus_iterator]))):
            counter_temp = 0
            for word_iterator in sentence_iterator.split():
              counter_temp = counter_temp +  RS[word_iterator]
            sentence_score[i] = counter_temp
          sorted_sentences_by_rank = dict(sorted(sentence_score.items(), reverse = True, key =lambda kv:(kv[1], kv[0])))
          # sorted_sentences_by_rank = sorted(sentence_score.items(),key=operator.itemgetter(1),reverse=True)
          rank_score = {}
          counter = 1;

          for sentence_index, score in sorted_sentences_by_rank.items():
            rank_score[sentence_index] = 1 - ((counter - 1)/len(sentence_score))
            counter = counter + 1
          if MAX_SENTENCE < len(sentence_score):
              for i in range(MAX_SENTENCE):
                for j in range(MAX_SENTENCE):
                  if(i==j):
                    continue;
                  else:
                    Wij[i][j] = Wij[i][j] * rank_score[i]

              for k in range(2):
                for i in range(MAX_SENTENCE):
                  for j in range(MAX_SENTENCE):
                    if(i==j):
                      continue;
                    else:
                      single_document_embedding[corpus_iterator][i].add(torch.mul(single_document_embedding[corpus_iterator][j], Wij[i][j]))
          else:
              for i in range(len(sentence_score)):
                for j in range(len(sentence_score)):
                  if(i==j):
                    continue;
                  else:
                    Wij[i][j] = Wij[i][j] * rank_score[i]

              for k in range(2):
                for i in range(len(sentence_score)):
                  for j in range(len(sentence_score)):
                    if(i==j):
                      continue;
                    else:
                      single_document_embedding[corpus_iterator][i].add(torch.mul(single_document_embedding[corpus_iterator][j], Wij[i][j]))

          # Normalised doc embedding
          doc_embedding = torch.mean(single_document_embedding[corpus_iterator],1).tolist()
          # doc_embedding = [float(sum(col))/len(col) for col in zip(*single_document_embedding[corpus_iterator])]
          doc_embedding_list.append(doc_embedding)
          corpus_iterator += 1

        out = self.fc(torch.tensor(doc_embedding_list))

        return out


# In[22]:


# Initialize network
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)


# In[23]:


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[24]:


loss_list=[]
# Train Network
for epoch in range(num_epochs):
  for itr_batch in range(0, len(df), batch_size):
    print(itr_batch, itr_batch+batch_size-1)
    t, all_doc_sent = get_all_sent_emb_in_doc(df.loc[itr_batch : itr_batch+batch_size-1], itr_batch)
    data = torch.as_tensor(t)
    targets = torch.as_tensor([i for i in df['Stance'][itr_batch : itr_batch+batch_size]])

  # Get data to cuda if possible
    data = data.to(device).squeeze(1)
    targets = targets.to(device)

    # forward
    scores = model(data.float()).to(device)
    
    loss = criterion(scores, targets).to(device)
    print("Epoch : ",epoch," Loss : ",loss.item())
    loss_list.append(loss)

    # backward
    optimizer.zero_grad()
    loss.backward()
    
    del data
    del targets
    del t
    del all_doc_sent
    del scores
    del loss
    # gradient descent or adam step
    optimizer.step()
  torch.save(model,'Trained_Model/ISOT_model'+str(epoch)+'.smdm')
del model


# In[ ]:


get_ipython().system('pip install matplotlib')

import matplotlib.pyplot as plt
plt.plot(range(len(loss_list)),[i.item() for i in loss_list], marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0,1)
plt.show()
torch.save(loss_list,'ISOT_loss.txt')

