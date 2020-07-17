from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.models import User,auth
from django.contrib import messages
from django.db import IntegrityError 
# Create your views here.
def login(request):
    
    return render(request,'login.html')

def checkcred(request):
    n=request.POST['name']
    p=request.POST['pass']
    import pymysql as pms
    con = pms.connect(user='root',password='root',host='localhost',db='healthanalysis')
    
    #Create a cursor
    cursor = con.cursor()
    
    #Create a query
    #query = "Insert into login(username,password) VALUES('"+uname+"','"+upass+"')"
    query = "Select * from userinfo where name = '"+n+"' and password = '"+p+"'"
    
    #Execute the query
    cursor.execute(query)
    
    #COmmit the operation
    con.commit()
    
    #CHeck success/failure
    count = cursor.rowcount
    #query = "Select * from login where username = '"+uname+"',password = '"+upass+"'"
    if count>0:
        return render(request,'home.html',{'name':n})
    else:
        messages.info(request,"Invalid Credentials")
        return render(request,'login.html')
    #return render(request,'home.html',{'name':n})
    

def newuser(request):
    return render(request,'signup.html')


def register(request):
    e=request.POST['email']
    n=request.POST['name']
    p=request.POST['pass']
    g=request.POST['gender']
    import pymysql as pms
    con = pms.connect(user='root',password='root',host='localhost',db='healthanalysis')
    
    #Create a cursor
    cursor = con.cursor()
    
    #Create a query
    query = "Insert into userinfo(name,email,password,gender) VALUES('"+n+"','"+e+"','"+p+"','"+g+"')"
    #query = "Select * from login where username = '"+uname+"' and password = '"+upass+"'"
    
    #Execute the query
    try:
        cursor.execute(query)
        con.commit()
    except IntegrityError:
        messages.info(request,"Error Occured")
    #COmmit the operation
    
    
    #CHeck success/failure
    count = cursor.rowcount
    #query = "Select * from login where username = '"+uname+"',password = '"+upass+"'"
    if count>0:
        messages.info(request,"User account created")
        return render(request,'home.html',{'name':n})
        #response = "Insert SUCCESS"
    else:
        messages.info(request,"Error Occured")
        return render(request,'signup.html')
    
    return HttpResponse(response)
    #return render(request,'home.html',{'name':n})
    
