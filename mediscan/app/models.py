from django.db import models
from django.contrib.auth.hashers import make_password

class Doctor(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]

    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    email = models.EmailField()
    country = models.CharField(max_length=100)
    phone_number = models.CharField(max_length=20)
    address = models.TextField()
    pincode = models.CharField(max_length=10)
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(auto_now=True)
    
    def save(self, *args, **kwargs):
        self.password = make_password(self.password)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name
    
class Patient(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]

    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    email = models.EmailField()
    country = models.CharField(max_length=100)
    phone_number = models.CharField(max_length=20)
    address = models.TextField()
    pincode = models.CharField(max_length=10)
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(auto_now=True)
    
    def save(self, *args, **kwargs):
        self.password = make_password(self.password)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name
    
class PatientRecord(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    address = models.CharField(max_length=255)
    contact = models.CharField(max_length=20)
    uid = models.CharField(max_length=50, unique=True)  # Unique identifier provided by the patient
    medical_conditions = models.TextField()
    medical_history = models.TextField()
    medication_history = models.TextField()
    allergies = models.TextField()
    test_results = models.TextField()
    hospitalizations = models.TextField()

    def __str__(self):
        return f"{self.name} ({self.uid})"
    
class DiseaseInformation(models.Model):
    name = models.CharField(max_length=100)
    treatment = models.TextField()

    def __str__(self):
        return self.name