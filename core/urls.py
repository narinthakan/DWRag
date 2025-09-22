from django.urls import path
from . import views

urlpatterns = [
    path("", views.home_view, name="home"),                  
    path("rag_query/", views.rag_query_view, name="rag_query"), 
    
]
