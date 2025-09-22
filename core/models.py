from django.db import models
from pgvector.django import VectorField

class MyDocument(models.Model):
    text = models.TextField()
    file = models.FileField(upload_to='documents/', blank=True, null=True)
    embedding = VectorField(dimensions=384)
    source = models.CharField(max_length=255)

    def __str__(self):
        return f"Document Chunk: {self.id}"