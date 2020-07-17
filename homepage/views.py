from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.models import User,auth
from django.contrib import messages
from django.db import IntegrityError

# Create your views here.
def home(request): 
    return render(request,'home.html')
def about(request):
    return render(request,'about.html')
def contact(request):
    return render(request,'contact.html')
def logout(request):
    return render(request,'login.html')
def diabetes(request):
    return render(request,'diabetes.html')
def heart(request):
    return render(request,'heart.html')
def breast(request):
    return render(request,'breastcancer.html')
def mental(request):
    return render(request,'mental.html')



def diabetescheck(request):
    
    l=[]
    pr=request.POST['pregnancy']
    l.append(pr)
    gl=request.POST['glucose']
    l.append(gl)
    bp=request.POST['bp']
    l.append(bp)
    sk=request.POST['skin']
    l.append(sk)
    ins=request.POST['insulin']
    l.append(ins)
    bmi=request.POST['bmi']
    l.append(bmi)
    dpf=request.POST['dpf']
    l.append(dpf)
    age=request.POST['age']
    l.append(age)
    
    import pandas as pd 
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv("A:\college tut\health analysis\dataset\diabetes.csv")
    X=df.loc[:, df.columns != 'Outcome']
    Y= df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=df['Outcome'], random_state=66)
    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, y_train)
    dict1 = {}
    i=0
    for columns in X.columns:
        if(columns=='BMI' or columns=='DiabetesPedigreeFunction'):
            temp = float(l[i])
        else:
            temp = int(l[i])
        i=i+1
        dict1[columns] = temp
    dict1          
    user_input = pd.DataFrame(dict1,index=[0],columns=X.columns)  
    Outcome = knn.predict(user_input)
    if(Outcome[0]==0):
        messages.info(request,"Probability is nill")
    else:
        messages.info(request,"Probability is positive")
    return render(request,'diabetes.html')



def heartcheck(request):
    l=[]
    a=request.POST['age']
    l.append(a)
    b=request.POST['gender']
    l.append(b)
    c=request.POST['angina']
    l.append(c)
    d=request.POST['bp']
    l.append(d)
    e=request.POST['cholestoral']
    l.append(e)
    f=request.POST['fbs']
    l.append(f)
    g=request.POST['electrocardiographic']
    l.append(g)
    h=request.POST['maxhr']
    l.append(h)
    i=request.POST['iangina']
    l.append(i)
    j=request.POST['depression']
    l.append(j)
    k=request.POST['slope']
    l.append(k)
    m=request.POST['majorves']
    l.append(m)
    n=request.POST['thal']
    l.append(n)
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.cm import rainbow


    # Other libraries
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Machine Learning
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    
    df=pd.read_csv("A:\college tut\health analysis\dataset\heart.csv")
    y = df['target']
    X = df.drop(['target'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
    knn_scores = []
    for k in range(1,21):
        knn_classifier = KNeighborsClassifier(n_neighbors = k)
        knn_classifier.fit(X_train, y_train)
        knn_scores.append(knn_classifier.score(X_test, y_test))
    knn = KNeighborsClassifier(n_neighbors=17)
    knn.fit(X_train, y_train)
    independent=df.columns
    independent=independent.delete(13)
    X=df[independent]
    dict1={}
    i=0
    for columns in X.columns:
        if(columns=='oldpeak'):
            temp=float(l[i])
        else:
            temp=int(l[i])
        dict1[columns]=temp
        i=i+1
    user_input=pd.DataFrame(dict1,index=[0],columns=X.columns)
    target=knn.predict(user_input)
    if(target[0]==0):
        messages.info(request,"Probability is nill")
    else:
        messages.info(request,"Probability is positive")

    return render(request,'heart.html')




def bcancercheck(request):
    l=[]
    a=request.POST['radius_mean']
    l.append(a)
    b=request.POST['texture_mean']
    l.append(b)
    c=request.POST['perimeter_mean']
    l.append(c)
    d=request.POST['area_mean']
    l.append(d)
    e=request.POST['concavity_mean']
    l.append(e)
    f=request.POST['concave_mean']
    l.append(f)
    g=request.POST['texture_se']
    l.append(g)
    h=request.POST['area_se']
    l.append(h)
    i=request.POST['texture_worst']
    l.append(i)
    j=request.POST['perimeter_worst']
    l.append(j)
    k=request.POST['area_worst']
    l.append(k)
    m=request.POST['concavity_worst']
    l.append(m)
    n=request.POST['concave_worst']
    l.append(n)
        
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    dataset=pd.read_csv("A:\college tut\health analysis\dataset\breast2.csv")
    X = dataset.iloc[:, 2:15].values 
    Y = dataset.iloc[:, 1].values 
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, Y_train)
    independent=dataset.columns
    independent=independent.delete(1)
    independent=independent.delete(0)
    X=dataset[independent]
    dict1={}
    i=0
    for columns in X.columns:
        dict1[columns]=l[i]
        i=i+1
    user_input=pd.DataFrame(dict1,index=[0],columns=X.columns)
    Outcome=forest.predict(user_input)
    if(Outcome[0]==0):
        messages.info(request,"Probability is Nill")
    else:
        messages.info(request,"Probability is Positive")
    return render(request,'breastcancer.html')



def mentalcheck(request):
    
    l=[]
    age=request.POST['age']
    l.append(age)
    gen=request.POST['gender']
    l.append(gen)
    fam=request.POST['family']
    l.append(fam)
    ben=request.POST['benefits']
    l.append(ben)
    care=request.POST['care']
    l.append(care)
    an=request.POST['anonymity']
    l.append(an)
    leave=request.POST['leave']
    l.append(leave)
    work=request.POST['work']
    l.append(work)
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
    
    train_df=pd.read_csv("A:\college tut\health analysis\dataset\mentalhealth.csv")
    train_df = train_df.drop(['Unnamed: 0'], axis= 1)
    
    labelDict = {}
    for feature in train_df:
        le = preprocessing.LabelEncoder()
        le.fit(train_df[feature])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        train_df[feature] = le.transform(train_df[feature])
        labelKey = 'label_' + feature
        labelValue = [le_name_mapping]
        labelDict[labelKey] =labelValue
        
    feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
    X = train_df[feature_cols]
    y = train_df.treatment
    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    from sklearn.neighbors import KNeighborsClassifier
    #knn = KNeighborsClassifier(n_neighbors=5)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    dict1 = {}
    i=0
    for columns in X.columns:
        temp = l[i]
        i=i+1
        dict1[columns] = temp
    dict1                                     
    #Create a dataframe using dict1                                 
    user_input = pd.DataFrame(dict1,index=[0],columns=X.columns)                   
    treatment = knn.predict(user_input)
    if(treatment[0]==0):
        messages.info(request,"Probability is nill")
    else:
        messages.info(request,"Probability is positive")
    return render(request,'mental.html')






    
