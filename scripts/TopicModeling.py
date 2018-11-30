#!/usr/bin/env python
# coding: utf-8

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pyLDAvis
import os
import glob 
import pyLDAvis.sklearn
# pyLDAvis.enable_notebook()
import pandas as pd
import json
import sys
import warnings
import nltk
# nltk.download('popular')
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

wnl = nltk.WordNetLemmatizer()

if not sys.warnoptions:
    warnings.simplefilter("ignore")

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

def getTokensSentence(sen):
    tokens = word_tokenize(sen)
    return tokens

def getPOSForWordnet(tag):
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

def fitAllLemmataInOneString(collectionOfBookTokens):
    fusionedStrings = []
    for book in collectionOfBookTokens:
        nString = ''
        for lemma in book: 
            nString = nString + ' '+lemma
        fusionedStrings.append(nString)
    return fusionedStrings

def setLDAParametters(nFeatures, nTopics,dataset):
    no_features = nFeatures
    # NMF is able to use tf-idf
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(dataset)
    # tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    no_topics = nTopics
    # Run NMF
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    nmf.fit(tfidf)
    return [nmf,tfidf,tfidf_vectorizer]

list_names = glob.glob('./data/simpletext/*.txt')
rawNotes = readAllFiles(list_names)
allNounsOfBooks = []
for book in rawNotes:
    allNounsOfBooks.append(getNounsFromFile(book))

nounBooksFusionedString = fitAllLemmataInOneString(allNounsOfBooks)

[nmfNoun,tfidMatrixNoun, tfidfVectorizerNoun] =setLDAParametters(50, 6,nounBooksFusionedString)

nmfNoun.n_components_
nmfNoun.components_
preparedData = pyLDAvis.sklearn.prepare(nmfNoun,tfidMatrixNoun, tfidfVectorizerNoun)

def getDocumentsPerTopic(listOfNameFiles, topicModel, matrixTF):
    topicsAndDocuments = []
    doc_topic = topicModel.transform(matrixTF)
    for n in range(doc_topic.shape[0]):
        topic_most_pr = doc_topic[n].argmax()
        
        topicsAndDocuments.append((listOfNameFiles[n],str(topic_most_pr)))
        # print("doc: {} topic: {}\n".format(n,topic_most_pr))
        dataFrameFileNameAndTopics = pd.DataFrame(data = topicsAndDocuments,columns=['NameDocument','Topic'])
    return dataFrameFileNameAndTopics

nameFromReadFiles = [name.split('/')[3] for name in list_names]
documentAndTopics = getDocumentsPerTopic(nameFromReadFiles,nmfNoun,tfidMatrixNoun)

# Format as JSON to be transferred to Node.js
d = preparedData.to_dict()['mdsDat']
preparedString = json.dumps(d)
jsonString = '{ \"prepared_data\":' + preparedString + ',' + '\"topics\":' + documentAndTopics.to_json() + '}'
# jsonString = '{ "prepared_data":' + '""' + ',' + '"topics":' + documentAndTopics.to_json() + '}'
# jsonString = '{ \"prepared_data\":' + '{ "i": "json!" }' + ',' + '\"topics\":' + '{ "j": "nosj!" }' + '}'

print(jsonString)