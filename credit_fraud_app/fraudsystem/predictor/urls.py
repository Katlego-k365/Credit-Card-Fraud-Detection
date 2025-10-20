from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload_file, name="upload_file"),
    path("view/", views.view_file, name="view_file"),
    path("predict/", views.predict, name="predict"),
    path("analyze/", views.analyze_file, name="analyze_file"),
    path("delete/", views.delete_file, name="delete_file"),
]
