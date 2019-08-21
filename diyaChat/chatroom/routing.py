from . import consumers
from django.urls import path

websocket_urlpatterns = [
    path('<str:lm_name>/', consumers.ChatConsumer),
]
