from django.db import models
from pgvector.django import VectorField
# Create your models here.

class Document(models.Model):
    file = models.FileField(upload_to='documents/')
    embedding = VectorField(dimensions=1536) # ควรตรงกับมิติของเวกเตอร์จากโมเดลของคุณ
  
