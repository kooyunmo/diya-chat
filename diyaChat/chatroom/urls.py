from django.urls import path
from . import views

from django.conf.urls import url

urlpatterns = [
   # ex) /chatroom/
   path('', views.index, name='index'),

   # ex) /chatroom/bert/
   #path('<str:lm_name>/', views.room, name='room'),

   url(r'^(?P<lm_name>[^/]+)/$', views.room, name='room'),
]
