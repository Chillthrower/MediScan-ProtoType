from django import forms
from .models import *

class DoctorSignupForm(forms.ModelForm):
    class Meta:
        model = Doctor
        fields = '__all__'
        
class PatientSignupForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = '__all__'
        
class PatientLoginForm(forms.Form):
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)

class DoctorLoginForm(forms.Form):
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)
    
class UploadImageForm(forms.Form):
    images = forms.FileField(label='Images', widget=forms.ClearableFileInput())
    
class PatientRecordForm(forms.ModelForm):
    class Meta:
        model = PatientRecord
        fields = '__all__'
        
class SymptomsForm(forms.Form):
    symptoms = forms.CharField(label='Symptoms', max_length=100)