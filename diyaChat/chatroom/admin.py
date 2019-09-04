from django.contrib import admin
from .models import Question, Room, UserInputDataset

# Register your models here.

admin.site.register(Question)
admin.site.register(Room)
admin.site.register(UserInputDataset)
