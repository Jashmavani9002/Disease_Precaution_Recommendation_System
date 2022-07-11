from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import io
from rest_framework.parsers import JSONParser
import os
import pandas as pd
import numpy as np
import pickle
import operator
import json
# Create your views here.

@api_view(['POST'])
def disease_prediction(request):
    try:
        if request.method == 'POST':
            json_data = request.body
            stream = io.BytesIO(json_data)
            pythondata = JSONParser().parse(stream)
            sym = pythondata["Symptoms"]

            res = {}
            sym_path = "/home/bibek/Downloads/Test/precaution_recomendation/data/Symptom-severity.csv"
            if os.path.exists(sym_path):
                df2 = pd.read_csv(sym_path)
                symptoms = list(set(df2['Symptom']))
                symptoms.sort()
                
                values = [0] * len(symptoms)
                temp = dict(zip(symptoms, values))
                err = []
            
                for i in sym:
                    if i.replace(" ","") in symptoms:
                       temp[i.replace(" ","")] = 1
                    else:
                        res={ "message": 'Symptoms not found'}
                        err.append(i)
                        res["Symptom"] = err
                        if len(err)==len(sym):
                            return Response(res)
            
                test = np.array(list(temp.values())).reshape(1,-1)
            
                doc_path = "/home/bibek/Downloads/Test/precaution_recomendation/models"
                if os.path.exists(doc_path):
            
                    if os.path.exists(doc_path+"/disease.sav"):
            
                        loaded_model = pickle.load(open(doc_path+'/disease.sav','rb'))
                        pred = loaded_model.predict_proba(test)
            
                        d = dict(enumerate(pred.flatten()))
                        sorted_d = dict(sorted(d.items(), key=operator.itemgetter(1),reverse=True))
                        ls = list(sorted_d.keys())[:5]
                        prob = np.multiply(np.array(list(sorted_d.values())[:5]),100)
            
                        json_path = "/home/bibek/Downloads/Test/precaution_recomendation/data"
                        if os.path.exists(json_path+"/disease.json"):
                            try :
                                f=open(json_path+"/disease.json","r")
                                data=json.load(f)
                                ind = list(data.keys())
                                val = list(data.values())
                                ans = {}
                                ans["Predicted_Disease"] = "Probability"
                                
                                if os.path.exists("/home/bibek/Downloads/Test/precaution_recomendation/data/symptom_precaution.csv"):
                                    df3 = pd.read_csv("/home/bibek/Downloads/Test/precaution_recomendation/data/symptom_precaution.csv")
                                    for i in range(len(ls)):
                                        ans[ind[val.index(ls[i])]] = prob[i]
                                    ans["Predicted precaution"] = list(df3[df3['Disease'] == ind[val.index(ls[0])]].set_index('Disease').T.to_dict('list').values())
            
                                    print(res)
                                    ans.update(res)
                                return Response(ans)
                            except Exception as esc:
                                res={"status": 404, "message": 'There is some error while predicting' }
                                return Response(res)
                    else:
                        res={"status": 404, "message": 'model is not found' }
                        return Response(res)
                else:
                    res={"status": 404, "message": 'There is some error in prediction' }
                    return Response(res)  
            else:
                res={"status": 404, "message": 'No data found' }
                return Response(res)
    except Exception as e:
        res={"status": 404, "message": "You are passing Data in incorrect format","Correct Format":{"Symptoms":["skin_rash","itching"]}}
        return Response(res)
