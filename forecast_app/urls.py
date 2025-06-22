from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_file, name='upload'),
    path('forecast/', views.forecast_view, name='forecast'),
    path('reports/', views.reports_view, name='reports'),
    path('settings/', views.settings_view, name='settings'),
]