from copy import copy
from distutils.log import error
from enum import unique
from importlib.resources import path
import math
from pathlib import Path
from click import open_file
import numpy as np
from collections import Counter
from tqdm import tqdm
from scipy.sparse import csr_matrix
import operator
from sklearn.preprocessing import normalize

class Document():
    Doc_Name = ""
    for iterator in range(1,6,1):
        Doc_Name = "Document_" + str(iterator) + ".txt"
        try:
            file = open(Doc_Name ,"x")
        except FileExistsError:
            continue


        
            # print("The file " + str(iterator) + " exists")
    # print("created" ,iterator+0, "Documents")
    file_1 = open (r"Document_1.txt", "w")
    file_1.write("At its founding in 1861, MIT was initially a small community of problem-solvers and science lovers eager to bring their knowledge to bear on the world.")
    file_2 = open (r"Document_2.txt", "w")
    file_2.write("Keanu Reeves accepted a lower salary in the Movie (The Devil's Advocate) so the producers could pay Al Pacino's asking price. When Pacino later heard about this, he donated the same amount of his salary to charity.")
    file_3 = open (r"Document_3.txt", "w")
    file_3.write("Microsoft Corporation is an American multinational technology corporation which produces computer software, consumer electronics, personal computers, and related services.")
    file_4 = open (r"Document_4.txt", "w")
    file_4.write("Antivirus software was originally developed to detect and remove computer viruses, hence the name. However, with the proliferation of other malware, antivirus software started to protect from other computer threats.")
    file_5 = open (r"Document_5.txt", "w")
    file_5.write("PDF is a powerful document which contains static elements (images and text), dynamic elements (forms) and embedded signatures. These elements are necessary to make document visually appealing and consistent, there is a darker side to it.")
            
    with open ('Document_1.txt', 'r', encoding='utf-8') as file_1:
        lines = file_1.readlines()
    with open ('Document_2.txt', 'r', encoding='utf-8') as file_2:
        lines = file_2.readlines()
    with open ('Document_3.txt', 'r', encoding='utf-8') as file_3:
        lines = file_3.readlines()
    with open ('Document_4.txt', 'r', encoding='utf-8') as file_4:
        lines = file_4.readlines()
    with open ('Document_5.txt', 'r', encoding='utf-8') as file_5:
        lines = file_5.readlines()
    
    files = Path('./Python').glob('*.txt')
    corpus = list()
    for file in files:
        corpus.append(file.read_text())

    def IDF(corpus , unique_words):
        idf_dict = {}
        N = len(corpus)
        for i in unique_words:
            count = 0
            for sen in corpus:
                if i in sen.split():
                    count = count+1
                idf_dict[i] = (math.log((1 + N) / (count+1))) + 1
        return idf_dict

    def fit(whole_data):
        unique_words = set()
        if isinstance(whole_data, (list,)):
            for x in whole_data:
                for y in x.split():
                    if len(y)<2:
                        continue
                    unique_words.add(y)
                unique_words = sorted(list(unique_words))
                vocab = {j:i for i,j in enumerate(unique_words)}
        Idf_values_of_all_unique_words = IDF(whole_data,unique_words)
        return vocab, Idf_values_of_all_unique_words
    vocabulary, idf_of_vocabulary = fit(corpus)
         
    #print(list(vocabulary.values()))

    def transform(dataset,vocabulary,idf_values):
        sparse_matrix = csr_matrix((len(dataset), len(vocabulary)), dtype=np.float64)
        for row in range(0,len(dataset)):
            number_of_words_in_sentence = Counter(dataset[row].split())
            for word in dataset[row].split():
                if word in list(vocabulary.keys()):
                    tf_idf_value=(number_of_words_in_sentence[word] / len(dataset[row].split()))*(idf_values[word])

                    sparse_matrix[row,vocabulary[word]] = tf_idf_value
            print('Norm Form\n', normalize(sparse_matrix, norm='12', axis=1,copy=True,return_norm=False))
            output = normalize(sparse_matrix, norm='12', axis=1,copy=True,return_norm=False)
            return output
    res = transform(Doc_Name,vocabulary,idf_of_vocabulary)
    print(sorted(res))