from django.http import HttpResponse
from django.shortcuts import render
from django.utils.safestring import mark_safe
from .models import Room

import json



def index(request):
    chatroom_list = Room.objects.order_by('lm_name')[:3]
    context = {
        'chatroom_list': chatroom_list
    }
    return render(request, 'chatroom/index.html', context)


def room(request, lm_name):
    context = {
        'lm_name': mark_safe(json.dumps(lm_name))
    }
    return render(request, 'chatroom/room.html', context)


def detail(request, lm_name):

    return HttpResponse("You're looking at chatroom using %s." % lm_name)

