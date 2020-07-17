from django.urls import path,include
from . import views

urlpatterns=[
    path('',views.login,name='login'),
    path('logincred',views.checkcred,name='checkcred'),
    path('newuser',views.newuser,name='newuser'),
    path('register',views.register,name='register'),
    
    path('',include('homepage.urls')),
]