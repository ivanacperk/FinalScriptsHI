#!/usr/bin/env python
# coding: utf-8

# In[2]:


#installation for pre-processing text
# pip install spacy


# In[ ]:


#installation for topic modeling visuals
pip install pyldavis


# In[ ]:


#installation of gensim -- library for unsupervised topic modeling and LDA modeling
pip install --upgrade gensim


# In[ ]:


#installation of regrex - syntax highlighting
pip install regrex


# In[ ]:


#installation for corpus
pip install corpora


# In[ ]:


#installation of natural language toolkit
pip install --user -U nltk


# In[1]:


import pandas as pd
import numpy as np

import re
import string

import spacy

import gensim
from gensim import corpora

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#data
ncontrols = pd.read_csv('sp800-53.csv')


# In[2]:


ncontrols.info()


# In[3]:


#creating variable for cleaning the text - removing any quotations/punctations and making everything lower case
def clean_text(text ): 
    delete_dict = {sp_character: '' for sp_character in string.punctuation} 
    delete_dict[' '] = ' ' 
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    return text1.lower()
    #print('cleaned:'+text1)


# In[10]:


import nltk
nltk.download('stopwords')


# In[4]:


ncontrols['Control Text'] = ncontrols['Control Text'].apply(clean_text)


# In[5]:


print(ncontrols['Control Text'])


# In[6]:


#pre-processing text - removing 'stopwords'
#stopwords = the, a, an,in (for example) do not carry significant meaning 
#it is important to remove stopwords b/c they are not meaningful - makes it easier to focus on important text
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
def remove_stopwords(text):
    textArr = text.split(' ')
    rem_text = " ".join([i for i in textArr if i not in stop_words])
    return rem_text

ncontrols['Control Text'] = ncontrols['Control Text'].apply(remove_stopwords)
print(ncontrols['Control Text'])


# In[7]:


#load spacy model
#lemma means root
#lemmatization is to reduce words to their original 'lemma' format so that they are easier to classify and 
#compare similar words
#for example = running, ran, run, runs
#finding similiar words for the nouns and adjectives
#create lemma for the nouns and adjectives

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def lemmatization(texts,allowed_postags=['NOUN', 'ADJ']): 
       output = []
       for sent in texts:
             doc = nlp(sent) 
             output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags ])
       return output


# In[8]:


#terms are being tokenized
#tokenization is important for text classification
#tokenization divides strings/sentences into small 'tokens' (smaller chunks) to seperate out the words 
#from the string/sentence
terms = ncontrols['Control Text'].tolist()
print(terms[1])
tokenized = lemmatization(terms)
print(tokenized[1])


# In[10]:


#create a dictionary
#create a document term matrix - this is a table of rows and colums that represents the text corpus
#every row is a document
#every column is a word that appears in the corpus
#the cells of the matrix count how many times each words appears in each document
#matrix is based on the dimension of the dictionary
dictionary = corpora.Dictionary(tokenized)
doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized]


# In[11]:


#create LDA model using gensim library
#genism is a python library for unsupervised topic modeling
LDA = gensim.models.ldamodel.LdaModel


# In[12]:


#building LDA model
#document matrix is what will be passing through
# chunksize determines the number of documents processed together in each iteration - 
#the bigger the chunksize the faster will speed up the training
#passes - controls how much we train the model on the corpus
#I chose to use 10 topics (number of topics can be underestimated and overestimated)
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=10, random_state=100,
                chunksize=1000, passes=50,iterations=100)


# In[13]:


#printing topics
#topic are = information, system, assignment, access, assignment, assignment, external, code, assignment, system
#noticed that it has assignment and system multiple times, this is due to LDA being a mixed membership model
#a mixed membership model (each document is assumed to be a mixture of numerous topics)
#each topic is followed by terms with the probability of being a representative or important to the topic 
lda_model.print_topics()


# In[14]:


# Visualize the topics
#https://github.com/bmabey/pyLDAvis
#https://speakerdeck.com/bmabey/visualizing-topic-models
#bubbles represent the topics - the bigger the bubble represents the frequencey of the topic in the csv file
pyLDAvis.enable_notebook()
visuals = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
visuals


# In[15]:


#measures for hod good the model is

#perplexity is the measure of how well a model predicts the text
# for perplexity, the lower the score the better (means the model has learned important topics)
print('\nPerplexity: ', lda_model.log_perplexity(doc_term_matrix,total_docs=10000))  

#coherence is making sense of the words or meaningful connections (understanding)
#the higher the score the better
from gensim.models.coherencemodel import CoherenceModel
coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized, dictionary=dictionary , coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[16]:


#now that I have baseline coherence score - can compute coherence values


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[17]:


#compute the conherence values by passing through the dictionary and document matrix
#took awhile to run
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=doc_term_matrix, texts=tokenized, start=2, limit=50, step=1)


# In[18]:


# present graph
#the less number of topics the greater the coherence score
#the more topics in a set the less understanding it will going to be
limit=50; start=2; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()# Print the coherence scores


# In[19]:


# Print the coherence scores
#number of topics equal to 2 had the greatest coherence value
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[20]:


# Select the model and print the topics
optimal_model = model_list[7]
model_topics = optimal_model.show_topics(formatted=False)
optimal_model.print_topics(num_words=10)


# In[21]:


# Visualize the topics with the optimal model
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(optimal_model, doc_term_matrix, dictionary)
vis


#  what did topic modeling show us?
#  - topic modeling ia great for displaying patterns or themes within documents
#  - those themes or patterns could be used as entities and the terms that follow the themes could be what need to be captured/identified (if glossary is not an option)
#  -
# 
