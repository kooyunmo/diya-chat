from django.shortcuts import render
from chatroom.models import Room

def index(request):
    chatroom_list = Room.objects.order_by('rank')[:5]
    context = {
        'chatroom_list': chatroom_list
    }
    return render(request, 'chatroom/index.html', context)
