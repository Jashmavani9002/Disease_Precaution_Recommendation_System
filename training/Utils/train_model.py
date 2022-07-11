
from importlib.resources import path
import pandas as pd
import numpy as np
import os
from rest_framework.response import Response
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
import json

def training():
    sym_path = "/home/bibek/Downloads/Test/precaution_recomendation/data/Symptom-severity.csv"
    dis_path = '/home/bibek/Downloads/Test/precaution_recomendation/data/dataset.csv'
    if os.path.exists(dis_path) and os.path.exists(sym_path):
        df1 = pd.read_csv(dis_path)
        df2 = pd.read_csv(sym_path)
    else:
        res={"status": 404, "message": 'No data found' }
        return Response(res)

    Disease = list(set(df1['Disease']))
    Disease.sort() # all disease
    symptoms = list(set(df2['Symptom']))
    symptoms.sort() # all symptoms

    encoded = []
    values = [0] * len(symptoms)
    for i in range(len(df1)):
        row = df1.iloc[i].values
        temp = dict(zip(symptoms, values))
        for i in row:
            if i is not np.nan and i not in Disease:
                temp[i.strip()] = 1

        encoded.append(temp)

    train = pd.DataFrame(data=encoded,columns=temp.keys())
    
    train = train.fillna(0)
    X_train = train
    
    dic = {str(j):i for i,j in enumerate(Disease)}
    y_train = []
    for i in df1['Disease'].values:
        y_train.append(dic[i])
    y_train = np.array(y_train)

    
    with open("/home/bibek/Downloads/Test/precaution_recomendation/data/disease.json", "w") as outfile:
        outfile.write(json.dumps(dic))
        outfile.close()
    # Model Training
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    
    doc_directory_path = "/home/bibek/Downloads/Test/precaution_recomendation/models"
    
    if os.path.exists(doc_directory_path):
        filename = doc_directory_path +'/disease.sav'
        vv=open(filename, 'wb')
        pickle.dump(classifier,vv)
        vv.close()  
    else:
        os.makedirs(doc_directory_path)
        filename = doc_directory_path +'/disease.sav'
        vv=open(filename, 'wb')
        pickle.dump(classifier,vv)
        vv.close() 
    