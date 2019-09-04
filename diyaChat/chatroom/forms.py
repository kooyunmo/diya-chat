from django import forms
from .models import UserInputDataset

class DatasetPostForm(forms.ModelForm):

    class Meta:
        model = UserInputDataset
        fields = ('question', 'answer',)
