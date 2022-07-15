# Disease_Precaution_Recommendation
Using django and djangorestframework.<br />
This project predict top 5 disease based on symptoms and recommend precaution related to disease.

# Project Directory

* /data/&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: All data files.
* /models/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Trained model save in this dir.
* /training/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Traininig model api.
* /prediction/&nbsp;: Prediction api.


# Packages

Django==3.2.7<br />
djangorestframework==3.12.3<br /> 
django-cors-headers <br />
numpy  <br />
pandas <br />
scikit-learn <br />
pickle4 <br />
requests<br />

<b> To install the above packages requirements.txt should be located in the project directory.</b>\
<b>Run command :
```bash 
pip install -r requirements.txt
```
<b>Alternate command :
 ```bash  
 pip3 install -r requirements.txt 
 ```
<b>Note</b>: To run the above commands , the python environment should be equipped with pip/pip3
scripts.

 ## Dataset
 
 ```bash
https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
```
 
 <p>
    <h3>Training</h3>
    <img src='assets/UI.PNG'>
    <br>
    <br>
    <h3>Prediction</h3>
    <img src='assets/UI1.PNG'>
</p>	

