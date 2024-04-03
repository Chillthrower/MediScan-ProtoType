from django.contrib import admin
from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('signupas/', views.signupas, name='signupas'),
    path('loginas/', views.loginas, name='loginas'),
    path('patient_signup/', views.patient_signup, name='patient_signup'),
    path('patient_login/', views.patient_login, name='patient_login'),
    path('doctor_signup/', views.doctor_signup, name='doctor_signup'),
    path('doctor_login/', views.doctor_login, name='doctor_login'),
    path('capture_or_upload/', views.capture_or_upload, name='capture_or_upload'),
    path('image_capture_doctor/', views.image_capture_doctor, name='image_capture_doctor'),
    path('image_upload_doctor/', views.image_upload_doctor, name='image_upload_doctor'),
    path('train_doctor/', views.train_model_doctor, name='train_model_doctor'),
    path('image_capture_patient/', views.image_capture_patient, name='image_capture_patient'),
    path('image_upload_patient/', views.image_upload_patient, name='image_upload_patient'),
    path('train_patient/', views.train_model_patient, name='train_model_patient'),
    path('record/', views.record, name='record'),
    path('add_patient_record/', views.add_patient_record, name='add_patient_record'),
    path('cool/', views.cool, name='cool'),
    path('doctor_capture_or_upload_patient/', views.doctor_capture_or_upload_patient, name='doctor_capture_or_upload_patient'),
    path('classify_image/', views.classify_image, name='classify_image'), 
    path('doctor_capture_patient/', views.doctor_capture_patient, name='doctor_capture_patient'),
    path('patient_cred/', views.patient_cred, name='patient_cred'),
    path('patient_cred_result/', views.patient_cred_result, name='patient_cred_result'),
    path('enter_uid_contact/', views.enter_uid_contact, name='enter_uid_contact'),
    path('doctor_classify_image/<str:uid>/<str:contact>/', views.doctor_classify_image, name='doctor_classify_image'),
    path('index/', views.index, name='index'),
    path('predict_disease/', views.predict_disease, name='predict_disease'),
    path('capture_or_upload_patient/', views.capture_or_upload_patient, name='capture_or_upload_patient'),
    path('patient_upload_email/', views.patient_upload_email, name='patient_upload_email'),
    path('login_capture_or_upload/', views.login_capture_or_upload, name='login_capture_or_upload'),
    path('image_login_upload_patient/', views.image_login_upload_patient, name='image_login_upload_patient'),
    path('image_login_capture_patient/', views.image_login_capture_patient, name='image_login_capture_patient'),
    path('patient_capture_email/', views.patient_capture_email, name='patient_capture_email'),
    path('doctor_upload_email/', views.doctor_upload_email, name='doctor_upload_email'),
    path('doctor_capture_email/', views.doctor_capture_email, name='doctor_capture_email'),
    path('image_login_upload_doctor/', views.image_login_upload_doctor, name='image_login_upload_doctor'),
    path('image_login_capture_doctor/', views.image_login_capture_doctor, name='image_login_capture_doctor'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)