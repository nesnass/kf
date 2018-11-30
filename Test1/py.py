#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# In[2]:


import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pyLDAvis
import os
import glob 
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
import pandas as pd


# In this first part, I only need to read the text without any preprocessing. One string represent the entire document. 

# In[3]:


list_name = glob.glob('./SimpleText/*.txt')

def readTxTfile(name_file):
    file = open(name_file, 'r')
    content = file.read()
    file.close()
    return content



# In[4]:


def readDataset(list_File_names):
    dataset = []
    for name in list_File_names:
        dataset.append(readTxTfile(name))
    return dataset


# In[5]:


simpleTextDataSet = readDataset(list_name)

# In[ ]:




# In[6]:


def setLDAParametters(nFeatures, nTopics,dataset):
    no_features = nFeatures
    # NMF is able to use tf-idf
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(dataset)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    no_topics = nTopics
    # Run NMF
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    nmf.fit(tfidf)
    return [nmf,tfidf,tfidf_vectorizer]
    


# In[7]:


[nmf,tfidMatrix, tfidfVectorizer] =setLDAParametters(50, 4,simpleTextDataSet)

# In[8]:


pyLDAvis.sklearn.prepare(nmf,tfidMatrix, tfidfVectorizer)

# ## In this second part, I will select only the nouns and I will lemmatize the text

# I need to have a string that represent an entired document. so if I pick only nouns I should tokenize, POS tag and lemmatize, select the POS and then concatenate all in one string 

# In[9]:


###The new libraries I need
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
wnl = nltk.WordNetLemmatizer()

# #### First, I put the new functions I need

# In[10]:


#read the files
def readTXTFile(nameFile):
    f = open(nameFile,'r')
    content = f.read().replace('\n',' ').split('.')
    return content

def readAllFiles(listOfFileNames):
    tmpdataset = []
    for nameFile in listOfFileNames:
        tmpdataset.append(readTXTFile(nameFile))
    return tmpdataset

def lemmatizeFile(fileContent):
    #fileContent is a list of sentences
    posSentences = [] 
    
    for sen in fileContent:

        posSentences.append(getPOSForFile(sen))
    
    wordNetLemmas = []
    for sen in posSentences:

        for tp in sen:
            posT = getPOSForWordnet(tp[1])
            if posT =='':
                wordNetLemmas.append([wnl.lemmatize(tp[0]),tp[0],tp[1]])
            else:
                wordNetLemmas.append([wnl.lemmatize(tp[0], pos=posT),tp[0],tp[1]])
        
    return wordNetLemmas

def getPOSForFile(sen):
    senTmp = getTokensSentence(sen)
    return nltk.pos_tag(senTmp)

#lemmatizeFile(rawNotes[0])
def getTokensSentence(sen):
    tokens = word_tokenize(sen)
    return tokens

def getPOSForWordnet(tag):
    #print tag[1]
    tagToReturn = ''
    if tag.startswith('J'):
        tagToReturn = wordnet.ADJ
    elif tag.startswith('V'):
        tagToReturn = wordnet.VERB
    elif tag in ['NN', 'NNS', 'NNP']:
        tagToReturn = wordnet.NOUN
    elif tag.startswith('R'):
        tagToReturn = wordnet.ADV
    else:
        tagToReturn = ''
    return tagToReturn



def getNounsFromFile(fileContent):
    nouns_lemma = []
    lemmaTokenPOS_file = lemmatizeFile(fileContent)
    for l_t_p in lemmaTokenPOS_file:
        if l_t_p[2] in ['NN', 'NNS', 'NNP']:
            nouns_lemma.append(l_t_p[0])
    return nouns_lemma

# In[11]:


def fitAllLemmataInOneString(collectionOfBookTokens):
    fusionedStrings = []
    for book in collectionOfBookTokens:
        nString = ''
        for lemma in book: 
            nString = nString + ' '+lemma
        fusionedStrings.append(nString)
    return fusionedStrings

# In[12]:


list_names = glob.glob('CorpusFrank/SimpleText/*.txt')
rawNotes = readAllFiles(list_names)
allNounsOfBooks = []
for book in rawNotes:
    allNounsOfBooks.append(getNounsFromFile(book))

nounBooksFusionedString = fitAllLemmataInOneString(allNounsOfBooks)

# In[13]:


[nmfNoun,tfidMatrixNoun, tfidfVectorizerNoun] =setLDAParametters(50, 6,nounBooksFusionedString)

# In[20]:


nmfNoun.n_components_

# In[22]:


nmfNoun.components_

# In[17]:


pyLDAvis.sklearn.prepare(nmfNoun,tfidMatrixNoun, tfidfVectorizerNoun)

# In[14]:


def getDocumentsPerTopic(listOfNameFiles, topicModel, matrixTF):
    topicsAndDocuments = []
    doc_topic = topicModel.transform(matrixTF)
    for n in range(doc_topic.shape[0]):
        topic_most_pr = doc_topic[n].argmax()
        
        topicsAndDocuments.append((listOfNameFiles[n],str(topic_most_pr)))
        #print("doc: {} topic: {}\n".format(n,topic_most_pr))
        dataFrameFileNameAndTopics = pd.DataFrame(data = topicsAndDocuments,columns=['NameDocument','Topic'])
    return dataFrameFileNameAndTopics

# In[15]:


#get the name of the files: 

nameFromReadFiles = [name.split('/')[2] for name in list_name]
documentAndTopics = getDocumentsPerTopic(nameFromReadFiles,nmfNoun,tfidMatrixNoun)

for topic,group in documentAndTopics.groupby("Topic"):
    print(topic)
    print(group)

# In[23]:


#This code is from 
#https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

# In[ ]:



