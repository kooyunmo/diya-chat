import datetime

from django.db import models
from django.utils import timezone

# Create your models here.

class Question(models.Model):
    question_text = models.CharField(max_length=200)
    date_time = models.DateTimeField(' question date time')

    def __str__(self):
        return self.question_text

    def was_entered_recently(self):
        return self.date_time >= timezone.now() - datetime.timedelta(days=1)

class Answer(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    answer_text = models.CharField(max_length=200)
    date_time = models.DateTimeField(' answer date time')

    def __str__(self):
        return self.answer_text

    def was_entered_recently(self):
        return self.date_time >= timezone.now() - datetime.timedelta(days=1)

class Room(models.Model):
    # ex) seq2seq, transformer, BERT
    lm_name = models.CharField(max_length=20)    # the chatbot language model adopted to chatroom

    def __str__(self):
        return self.lm_name
