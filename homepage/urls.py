from django.urls import path
from . import views

urlpatterns=[
    path('home/',views.home,name='home'),
    path('about/',views.about,name='about'),
    path('contact/',views.contact,name='contact'),
    path('',views.logout,name='logout'),
    path('diabetes/',views.diabetes,name='diabetes'),
    path('heart/',views.heart,name='heart'),
    path('breast/',views.breast,name='breast'),
    path('mental/',views.mental,name='mental'),
    path('diabetes/diabetescheck',views.diabetescheck,name='diabetescheck'),
    path('heart/heartcheck',views.heartcheck,name='heartcheck'),
     path('breast/bcancercheck',views.bcancercheck,name='bcancercheck'),
    path('mental/mentalcheck',views.mentalcheck,name='mentalcheck'),


]