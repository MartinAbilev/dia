from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
        path('generate-audio/', views.generate_audio_view, name='generate_audio'),

]
