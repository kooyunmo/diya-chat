from django.urls import path
from . import views

from django.conf.urls import url

app_name = 'chatroom'
urlpatterns = [
   # ex) /chatroom/
   path('', views.index, name='index'),

   # ex) /chatroom/bert/
   path('<str:lm_name>/', views.room, name='room'),

   #path('<str:lm_name>/<str:num>/', views.helloworld, name='helloworld'),

   path('<str:lm_name>/<str:message>/', views.message, name='message'),

]
