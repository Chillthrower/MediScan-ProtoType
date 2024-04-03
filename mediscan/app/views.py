from django.shortcuts import render, redirect, get_object_or_404
from .forms import *
from .models import *
from django.http import JsonResponse, HttpResponseRedirect
from django.contrib import messages
from django.contrib.auth.hashers import check_password
from django.contrib.auth import authenticate, login
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from keras.preprocessing.image import load_img, img_to_array
import imghdr
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras import Sequential
import os
import shutil
from django.conf import settings
from .prediction import predictDisease
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from django.urls import reverse
from tensorflow.keras.preprocessing import image
import os
import base64
from PIL import Image
import torch
from torchvision import transforms
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from .models import Patient, PatientRecord
from facenet_pytorch import InceptionResnetV1  # Assuming your model is defined in app/models.py
from django.conf import settings
import torch
from torchvision.transforms import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from io import BytesIO

# Create your views here.

def home(request):
    user = request.user
    return render(request, 'app/home.html', {'user': user})

def cool(request):
    return render(request, 'app/doctor/cool.html')

def signupas(request):
    return render(request, 'app/signupas.html')

def capture_or_upload(request):
    name = request.GET.get('name', '')
    print(name)
    return render(request, 'app/capture_or_upload.html', {'name': name})

def login_capture_or_upload(request):
    email = request.GET.get('email', '')
    return render(request, 'app/login_capture_or_upload.html', {'email': email})

def loginas(request):
    return render(request, 'app/loginas.html')

def patient_cred(request):
    form = PatientRecordForm()
    return render(request, 'app/patient/patient_cred.html', {'form': form})

def patient_cred_result(request):
    return render(request, 'app/patient/patient_cred_result.html')

def record(request):
    return render(request, 'app/doctor/record.html')

def doctor_capture_or_upload_patient(request):
    return render(request, 'app/doctor/doctor_capture_or_upload_patient.html')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

######################################################   SIGN UP   #############################################################

def patient_signup(request):
    if request.method == 'POST':
        form = PatientSignupForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            form.save()
            return HttpResponseRedirect(reverse('capture_or_upload') + f'?user_type=patient&name={name}') #/capture_or_upload/?user_type=patient&name={name}
        else:
            messages.error(request, "Form is not valid.")
    else:
        form = PatientSignupForm()
    return render(request, 'app/patient/patient_signup.html', {'form': form})

def doctor_signup(request):
    if request.method == 'POST':
        form = DoctorSignupForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            form.save()
            return HttpResponseRedirect(reverse('capture_or_upload') + f'?user_type=doctor&name={name}')
        else:
            messages.error(request, "Form is not valid.")
    else:
        form = DoctorSignupForm()
    return render(request, 'app/doctor/doctor_signup.html', {'form': form})

######################################################   LOG IN   #############################################################

def patient_login(request):
    if request.method == 'POST':
        form = PatientLoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            try:
                user = Patient.objects.get(email=email)
                if check_password(password, user.password):
                    login(request, user)
                    # return redirect('patient_cred')
                    return HttpResponseRedirect(reverse('login_capture_or_upload') + f'?user_type=patient&email={email}')
                else:
                    messages.error(request, 'Invalid email or password. Please try again.')
            except Patient.DoesNotExist:
                messages.error(request, 'Invalid email or password. Please try again.')
    else:
        form = PatientLoginForm()
    return render(request, 'app/patient/patient_login.html', {'form': form})

# def doctor_login(request):
#     if request.method == 'POST':
#         form = DoctorLoginForm(request.POST)
#         if form.is_valid():
#             email = form.cleaned_data['email']
#             password = form.cleaned_data['password']
#             try:
#                 user = Doctor.objects.get(email=email)
#                 if check_password(password, user.password):
#                     login(request, user)
#                     return redirect('record')
#                 else:
#                     messages.error(request, 'Invalid email or password. Please try again.')
#             except Doctor.DoesNotExist:
#                 messages.error(request, 'Invalid email or password. Please try again.')
#     else:
#         form = DoctorLoginForm()
#     return render(request, 'app/doctor/doctor_login.html', {'form': form})

def doctor_login(request):
    if request.method == 'POST':
        form = DoctorLoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            try:
                user = Doctor.objects.get(email=email)
                if check_password(password, user.password):
                    login(request, user)
                    # return redirect('patient_cred')
                    return HttpResponseRedirect(reverse('login_capture_or_upload') + f'?user_type=doctor&email={email}')
                else:
                    messages.error(request, 'Invalid email or password. Please try again.')
            except Doctor.DoesNotExist:
                messages.error(request, 'Invalid email or password. Please try again.')
    else:
        form = DoctorLoginForm()
    return render(request, 'app/doctor/doctor_login.html', {'form': form})

######################################################   IMAGE CAPTURE   #############################################################


def image_capture_doctor(request):
    if request.method == 'POST' and 'capture' in request.POST:
        name = request.GET.get('name', '')
        print("Name:", name)
        
        frame = request.POST.get('image_data')
        img_data = frame.split(",")[1]
        img_data = bytes(img_data, 'utf-8')

        if not name:
            return render(request, 'home.html', {'error_message': 'Please provide your full name'})


        media_dir = os.path.join('media/doctor', name)
        if not os.path.exists(media_dir):
            os.makedirs(media_dir)


        image_number = 1
        while os.path.exists(os.path.join(media_dir, f'captured_image_{image_number}.jpg')):
            image_number += 1


        with open(os.path.join(media_dir, f'captured_image_{image_number}.jpg'), 'wb') as f:
            f.write(base64.b64decode(img_data))
        print("Image captured successfully!")
        return render(request, 'app/doctor/image_capture_doctor.html', {'success_message': 'Image captured successfully!'})

    return render(request, 'app/doctor/image_capture_doctor.html')

def image_capture_patient(request):
    if request.method == 'POST' and 'capture' in request.POST:
        name = request.GET.get('name', '')
        print("Name:", name)
        
        frame = request.POST.get('image_data')
        img_data = frame.split(",")[1]
        img_data = bytes(img_data, 'utf-8')

        if not name:
            return render(request, 'home.html', {'error_message': 'Please provide your full name'})

        media_dir = os.path.join('media/patient', name)
        if not os.path.exists(media_dir):
            os.makedirs(media_dir)

        image_number = 1
        while os.path.exists(os.path.join(media_dir, f'captured_image_{image_number}.jpg')):
            image_number += 1

        with open(os.path.join(media_dir, f'captured_image_{image_number}.jpg'), 'wb') as f:
            f.write(base64.b64decode(img_data))
        print("Image captured successfully!")
        return render(request, 'app/patient/image_capture_patient.html', {'success_message': 'Image captured successfully!'})

    return render(request, 'app/patient/image_capture_patient.html')

######################################################   IMAGE UPLOAD   #############################################################

def image_upload_doctor(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            name = request.GET.get('name', '')
            if not name:
                return render(request, 'image_upload/image_upload.html', {'form': form, 'error_message': 'Please provide your full name'})

            media_dir = os.path.join('media/doctor', name)
            if not os.path.exists(media_dir):
                os.makedirs(media_dir)

            for uploaded_image in request.FILES.getlist('images'):
                image_number = 1
                while os.path.exists(os.path.join(media_dir, f'uploaded_image_{image_number}.jpg')):
                    image_number += 1
                with open(os.path.join(media_dir, f'uploaded_image_{image_number}.jpg'), 'wb') as f:
                    for chunk in uploaded_image.chunks():
                        f.write(chunk)
            return render(request, 'app/doctor/image_upload_doctor.html', {'form': form, 'success_message': 'Images uploaded successfully!'})

    else:
        form = UploadImageForm()
    return render(request, 'app/doctor/image_upload_doctor.html', {'form': form})

def image_upload_patient(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            name = request.GET.get('name', '')
            if not name:
                return render(request, 'image_upload/image_upload.html', {'form': form, 'error_message': 'Please provide your full name'})

            media_dir = os.path.join('media/patient', name)
            if not os.path.exists(media_dir):
                os.makedirs(media_dir)

            for uploaded_image in request.FILES.getlist('images'):
                image_number = 1
                while os.path.exists(os.path.join(media_dir, f'uploaded_image_{image_number}.jpg')):
                    image_number += 1
                with open(os.path.join(media_dir, f'uploaded_image_{image_number}.jpg'), 'wb') as f:
                    for chunk in uploaded_image.chunks():
                        f.write(chunk)
            return render(request, 'app/patient/image_upload_patient.html', {'form': form, 'success_message': 'Images uploaded successfully!'})

    else:
        form = UploadImageForm()
    return render(request, 'app/patient/image_upload_patient.html', {'form': form})

######################################################   MODAL TRAINING   #############################################################

def train_model_doctor(request):
    # images_folder = 'G:/MEDISCAN_TEST04/MEDISCAN/MEDIA/DOCTOR'
    images_folder = os.path.join(settings.MEDIA_ROOT, 'doctor')
    data_dir = os.path.join(settings.BASE_DIR, 'media', 'doctor')

    image_exts = ['jpeg', 'jpg', 'bmp', 'png']
    for image_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e:
                print('Issue with image {}'.format(image_path))

    input_path = []
    label = []

    for class_name in os.listdir(images_folder):
        class_path = os.path.join(images_folder, class_name)
        for path in os.listdir(class_path):
            input_path.append(os.path.join(images_folder, class_name, path))
            label.append(class_name)

    df = pd.DataFrame({'images': input_path, 'label': label})

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.5, 1.5],
        channel_shift_range=50,
        vertical_flip=True,
    )

    val_generator = ImageDataGenerator(rescale=1. / 255)

    train_iterator = train_generator.flow_from_dataframe(
        train,
        x_col='images',
        y_col='label',
        target_size=(128, 128),
        batch_size=32,
        class_mode='sparse'  # Use 'sparse' for integer labels
    )

    val_iterator = val_generator.flow_from_dataframe(
        test,
        x_col='images',
        y_col='label',
        target_size=(128, 128),
        batch_size=32,
        class_mode='sparse'  # Use 'sparse' for integer labels
    )

    num_classes = len(df['label'].unique())  # Number of unique classes in the dataset

    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(train_iterator, epochs=5, validation_data=val_iterator)

    model.save('DOCTOR.h5')

    return redirect("doctor_login")


def train_model_patient(request):
    # images_folder = 'G:/MEDISCAN_TEST04/MEDISCAN/MEDIA/DOCTOR'
    images_folder = os.path.join(settings.MEDIA_ROOT, 'patient')
    data_dir = os.path.join(settings.BASE_DIR, 'media', 'patient')

    image_exts = ['jpeg', 'jpg', 'bmp', 'png']
    for image_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e:
                print('Issue with image {}'.format(image_path))

    input_path = []
    label = []

    for class_name in os.listdir(images_folder):
        class_path = os.path.join(images_folder, class_name)
        for path in os.listdir(class_path):
            input_path.append(os.path.join(images_folder, class_name, path))
            label.append(class_name)

    df = pd.DataFrame({'images': input_path, 'label': label})

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.5, 1.5],
        channel_shift_range=50,
        vertical_flip=True,
    )

    val_generator = ImageDataGenerator(rescale=1. / 255)

    train_iterator = train_generator.flow_from_dataframe(
        train,
        x_col='images',
        y_col='label',
        target_size=(128, 128),
        batch_size=32,
        class_mode='sparse'  # Use 'sparse' for integer labels
    )

    val_iterator = val_generator.flow_from_dataframe(
        test,
        x_col='images',
        y_col='label',
        target_size=(128, 128),
        batch_size=32,
        class_mode='sparse'  # Use 'sparse' for integer labels
    )

    num_classes = len(df['label'].unique())  # Number of unique classes in the dataset

    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(train_iterator, epochs=2, validation_data=val_iterator)

    model.save('PATIENT.h5')

    return redirect("patient_login")

######################################################   DOCTOR   #############################################################

def enter_uid_contact(request):
    if request.method == 'POST':
        uid = request.POST.get('uid')
        contact = request.POST.get('contact')

        return redirect('doctor_classify_image', uid=uid, contact=contact)

    return render(request, 'app/doctor/enter_uid_contact.html')

def add_patient_record(request):
    if request.method == 'POST':
        form = PatientRecordForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('cool')
    else:
        form = PatientRecordForm()
    return render(request, 'app/doctor/add_patient_record.html', {'form': form})

def doctor_classify_image(request, uid, contact):
    model_path = os.path.join(settings.BASE_DIR, 'YOLO.h5')
    model = load_model(model_path)

    input_image = preprocess_image(os.path.join(settings.MEDIA_ROOT, 'uploads', 'captured_image.jpg'))
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)

    data_dir = os.path.join(settings.BASE_DIR, 'media', 'patient')
    class_labels = sorted(os.listdir(data_dir))
    predicted_class_label = class_labels[predicted_class_index]

    patient_records = PatientRecord.objects.filter(name=predicted_class_label, uid=uid, contact=contact)

    context = {
        'image_path': os.path.join(settings.MEDIA_URL, 'uploads', 'captured_image.jpg'),
        'predicted_class_label': predicted_class_label,
        'patient_records': patient_records
    }
    return render(request, 'app/doctor/result.html', context)


def doctor_capture_patient(request):
    if request.method == 'POST':
        frame = request.POST.get('image_data')
        img_data = frame.split(",")[1]
        img_data = bytes(img_data, 'utf-8')

        media_dir = os.path.join('media/uploads')
        if not os.path.exists(media_dir):
            os.makedirs(media_dir)

        with open(os.path.join(media_dir, 'captured_image.jpg'), 'wb') as f:
            f.write(base64.b64decode(img_data))
        
        return redirect('enter_uid_contact')  # Assuming 'enter_uid_contact' is the URL name for the desired page

    return render(request, 'app/doctor/doctor_capture_patient.html')

# def classify_image(request):
#     if request.method == 'POST' and request.FILES['image']:
#         uploaded_image = request.FILES['image']
#         uid = request.POST.get('uid', '')  # How does this work
#         contact = request.POST.get('contact', '')
        
#         upload_folder = os.path.join(settings.MEDIA_ROOT, 'uploads')
#         if os.path.exists(upload_folder):
#             shutil.rmtree(upload_folder)
            
#         os.makedirs(upload_folder)
        
#         input_image_path = os.path.join(upload_folder, uploaded_image.name)
#         print(input_image_path)
        
#         with open(input_image_path, 'wb+') as destination:
#             for chunk in uploaded_image.chunks():
#                 destination.write(chunk)

#         model_path = os.path.join(settings.BASE_DIR, 'YOLO.h5')
#         model = load_model(model_path)

#         input_image = preprocess_image(input_image_path)
#         predictions = model.predict(input_image)
#         predicted_class_index = np.argmax(predictions)

#         data_dir = os.path.join(settings.BASE_DIR, 'media', 'patient')
#         class_labels = sorted(os.listdir(data_dir))
#         predicted_class_label = class_labels[predicted_class_index]

#         if uid and contact:
#             patient_records = PatientRecord.objects.filter(name=predicted_class_label, uid=uid, contact=contact)
#         elif uid:
#             patient_records = PatientRecord.objects.filter(name=predicted_class_label, uid=uid)
#         elif contact:
#             patient_records = PatientRecord.objects.filter(name=predicted_class_label, contact=contact)
#         else:
#             patient_records = PatientRecord.objects.filter(name=predicted_class_label)

#         context = {
#             'image_path': os.path.join(settings.MEDIA_URL, 'uploads', uploaded_image.name),
#             'predicted_class_label': predicted_class_label,
#             'patient_records': patient_records
#         }
#         return render(request, 'app/doctor/result.html', context)

#     return render(request, 'app/doctor/doctor_upload_patient.html')

def classify_image(request):
    uid = request.POST.get('uid', '')  # How does this work
    contact = request.POST.get('contact', '')
    if request.method == 'POST' and request.FILES.get('image'):
        model_path = os.path.join(settings.BASE_DIR, 'trained_model.pt')
        model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=8)
        model.load_state_dict(torch.load("trained_model.pt", map_location=torch.device('cpu')))
        
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image_path = request.FILES['image']
        upload_folder = os.path.join(settings.MEDIA_ROOT, 'uploads')
        if os.path.exists(upload_folder):
            shutil.rmtree(upload_folder)
            
        os.makedirs(upload_folder)
        
        input_image_path = os.path.join(upload_folder, image_path.name)
        print(input_image_path)
        
        with open(input_image_path, 'wb+') as destination:
            for chunk in image_path.chunks():
                destination.write(chunk)
        image = Image.open(image_path)
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_batch = input_batch.to(device)
        
        with torch.no_grad():
            output = model(input_batch)

        # predictions = model.predict(input_image)
        # predicted_class_index = predictions.argmax()
        predicted_class_index = torch.argmax(output, dim=1).item()

        data_dir = os.path.join(settings.BASE_DIR, 'media', 'patient')
        class_labels = sorted(os.listdir(data_dir))
        predicted_class_label = class_labels[predicted_class_index]
        print(predicted_class_label)

        if uid and contact:
            patient_records = PatientRecord.objects.filter(name=predicted_class_label, uid=uid, contact=contact)
        elif uid:
            patient_records = PatientRecord.objects.filter(name=predicted_class_label, uid=uid)
        elif contact:
            patient_records = PatientRecord.objects.filter(name=predicted_class_label, contact=contact)
        else:
            patient_records = PatientRecord.objects.filter(name=predicted_class_label)

        context = {
            'image_path': os.path.join(settings.MEDIA_URL, 'uploads', image_path.name),
            'predicted_class_label': predicted_class_label,
            'patient_records': patient_records
        }
        return render(request, 'app/doctor/result.html', context)

    return render(request, 'app/doctor/doctor_upload_patient.html')

def preprocess_input_image(image_file):
    # Read the content of the uploaded file
    image_content = image_file.read()
    # Create a BytesIO object with the content
    image_stream = BytesIO(image_content)
    # Load the image from the BytesIO object
    img = image.load_img(image_stream, target_size=(128, 128))
    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)
    # Expand the dimensions to match the expected input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the pixel values to the range [0, 1]
    img_array /= 255.0
    return img_array

def doctor_upload_email(request):
    email = request.GET.get('email', '')
    if request.method == 'POST' and request.FILES.get('image'):
        model_path = os.path.join(settings.BASE_DIR, 'trained_model.pt')
        model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=8)
        model.load_state_dict(torch.load("trained_model.pt", map_location=torch.device('cpu')))
        
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image_path = request.FILES['image']
        image = Image.open(image_path)
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_batch = input_batch.to(device)
        
        with torch.no_grad():
            output = model(input_batch)

        # predictions = model.predict(input_image)
        # predicted_class_index = predictions.argmax()
        predicted_class_index = torch.argmax(output, dim=1).item()

        data_dir = os.path.join(settings.BASE_DIR, 'media', 'patient')
        class_labels = sorted(os.listdir(data_dir))
        predicted_class_label = class_labels[predicted_class_index]
        print(predicted_class_label)

        doctor = get_object_or_404(Doctor, name=predicted_class_label)
        if doctor.email == email:
            return redirect('record')
        else:
            return JsonResponse({'result': 'Email does not match'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

def image_login_upload_doctor(request):
    email = request.GET.get('email', '')
    return render(request, 'app/doctor/image_login_upload_doctor.html', {'email': email})

def image_login_capture_doctor(request):
    email = request.GET.get('email', '')
    return render(request, 'app/doctor/image_login_capture_doctor.html', {'email': email})\
        
def preprocess_input1_image(img):
    img = img.resize((128, 128))  # Resize image to desired dimensions
    img_array = np.array(img)  # Convert PIL Image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def doctor_capture_email(request):
    email = request.GET.get('email', '')
    if request.method == 'POST':
        frame = request.POST.get('image_data')
        img_data = frame.split(",")[1]
        img_data = bytes(img_data, 'utf-8')

        media_dir = os.path.join('media/capture')
        if not os.path.exists(media_dir):
            os.makedirs(media_dir)

        with open(os.path.join(media_dir, 'captured_image.jpg'), 'wb') as f:
            f.write(base64.b64decode(img_data))
            
        model_path = os.path.join(settings.BASE_DIR, 'trained_model.pt')
        model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=8)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_path = os.path.join(settings.MEDIA_ROOT, 'capture', 'captured_image.jpg')
        image = Image.open(image_path)
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_batch = input_batch.to(device)
        
        with torch.no_grad():
            output = model(input_batch)

        predicted_class_index = torch.argmax(output, dim=1).item()

        data_dir = os.path.join(settings.BASE_DIR, 'media', 'patient')
        class_labels = sorted(os.listdir(data_dir))
        predicted_class_label = class_labels[predicted_class_index]
        print("Predicted Class Label:", predicted_class_label)

        doctor = get_object_or_404(Doctor, name=predicted_class_label)
        if doctor.email == email:
            return redirect('record')
        else:
            return JsonResponse({'result': 'Email does not match'}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)


######################################################   PREDICTION   #############################################################

def index(request):
    return render(request, 'app/index.html')

@csrf_exempt
def predict_disease(request):
    if request.method == 'POST' and request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
        symptoms = request.POST.getlist('symptoms[]')
        predictions = predictDisease(symptoms)
        
        predicted_disease_name = predictions['final_prediction']
        disease_information = DiseaseInformation.objects.filter(name=predicted_disease_name).first()
        if disease_information:
            treatment = disease_information.treatment
            predictions['treatment'] = treatment
        else:
            predictions['treatment'] = 'Treatment information not available.'
        
        return JsonResponse(predictions)
    else:
        return JsonResponse({'error': 'Invalid request'})

######################################################   PATIENTS   #############################################################

def capture_or_upload_patient(request):
    return render(request, 'app/patient/capture_or_upload_patient.html')

def preprocess_input_image(image_file):
    # Read the content of the uploaded file
    image_content = image_file.read()
    # Create a BytesIO object with the content
    image_stream = BytesIO(image_content)
    # Load the image from the BytesIO object
    img = image.load_img(image_stream, target_size=(128, 128))
    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)
    # Expand the dimensions to match the expected input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the pixel values to the range [0, 1]
    img_array /= 255.0
    return img_array

# def doctor_upload_email(request):
#     email = request.GET.get('email', '')
#     if request.method == 'POST' and request.FILES.get('image'):
#         model_path = os.path.join(settings.BASE_DIR, 'YOLO.h5')
#         model = load_model(model_path)

#         input_image = preprocess_input_image(request.FILES['image'])

#         predictions = model.predict(input_image)
#         predicted_class_index = predictions.argmax()

#         data_dir = os.path.join(settings.BASE_DIR, 'media', 'patient')
#         class_labels = sorted(os.listdir(data_dir))
#         predicted_class_label = class_labels[predicted_class_index]
#         print(predicted_class_label)

#         patient = get_object_or_404(Patient, name=predicted_class_label)
#         if patient.email == email:
#             patient = Patient.objects.get(name=predicted_class_label, email=email)
#             patient_records = PatientRecord.objects.filter(name=predicted_class_label)
#             return render(request, 'app/patient/patient_cred_result.html', {'patient_records': patient_records})
#         else:
#             return JsonResponse({'result': 'Email does not match'}, status=400)

#     return JsonResponse({'error': 'Invalid request method'}, status=405)

def patient_upload_email(request):
    email = request.GET.get('email', '')
    if request.method == 'POST' and request.FILES.get('image'):
        model_path = os.path.join(settings.BASE_DIR, 'trained_model.pt')
        model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=8)
        model.load_state_dict(torch.load("trained_model.pt", map_location=torch.device('cpu')))
        
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image_path = request.FILES['image']
        image = Image.open(image_path)
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_batch = input_batch.to(device)
        
        with torch.no_grad():
            output = model(input_batch)

        # predictions = model.predict(input_image)
        # predicted_class_index = predictions.argmax()
        predicted_class_index = torch.argmax(output, dim=1).item()

        data_dir = os.path.join(settings.BASE_DIR, 'media', 'patient')
        class_labels = sorted(os.listdir(data_dir))
        predicted_class_label = class_labels[predicted_class_index]
        print(predicted_class_label)

        patient = get_object_or_404(Patient, name=predicted_class_label)
        if patient.email == email:
            patient = Patient.objects.get(name=predicted_class_label, email=email)
            patient_records = PatientRecord.objects.filter(name=predicted_class_label)
            return render(request, 'app/patient/patient_cred_result.html', {'patient_records': patient_records})
        else:
            return JsonResponse({'result': 'Email does not match'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

def image_login_upload_patient(request):
    email = request.GET.get('email', '')
    return render(request, 'app/patient/image_login_upload_patient.html', {'email': email})

def image_login_capture_patient(request):
    email = request.GET.get('email', '')
    return render(request, 'app/patient/image_login_capture_patient.html', {'email': email})

def image_login_capture_patient(request):
    email = request.GET.get('email', '')
    return render(request, 'app/patient/image_login_capture_patient.html', {'email': email})

def preprocess_input1_image(img):
    img = img.resize((128, 128))  # Resize image to desired dimensions
    img_array = np.array(img)  # Convert PIL Image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# def doctor_capture_email(request):
#     email = request.GET.get('email', '')
#     if request.method == 'POST' and request.POST.get('image_data'):
#         model_path = os.path.join(settings.BASE_DIR, 'YOLO.h5')
#         model = load_model(model_path)

#         frame = request.POST.get('image_data')
#         img_data = frame.split(",")[1]
#         img_data = base64.b64decode(img_data)
        
#         img = Image.open(BytesIO(img_data))

#         input_image = preprocess_input1_image(img)

#         predictions = model.predict(input_image)
#         predicted_class_index = predictions.argmax()

#         data_dir = os.path.join(settings.BASE_DIR, 'media', 'patient')
#         class_labels = sorted(os.listdir(data_dir))
#         predicted_class_label = class_labels[predicted_class_index]
#         print(predicted_class_label)
        
#         patient = get_object_or_404(Patient, name=predicted_class_label)

#         if patient.email == email:
#             patient = Patient.objects.get(name=predicted_class_label, email=email)
#             patient_records = PatientRecord.objects.filter(name=predicted_class_label)
#             return render(request, 'app/patient/patient_cred_result.html', {'patient_records': patient_records})
#         else:
#             return JsonResponse({'result': 'Email does not match'}, status=400)

#     return JsonResponse({'error': 'Invalid request method'}, status=405)

# def doctor_capture_email(request):
#     email = request.GET.get('email', '')
#     if request.method == 'POST' and 'capture' in request.POST:
#         frame = request.POST.get('image_data')
#         img_data = frame.split(",")[1]
#         img_data = bytes(img_data, 'utf-8')

#         media_dir = os.path.join('media/capture')
#         if not os.path.exists(media_dir):
#             os.makedirs(media_dir)

#         with open(os.path.join(media_dir, 'captured_image.jpg'), 'wb') as f:
#             f.write(base64.b64decode(img_data))
#         print("Image captured successfully!")

#         # Model Prediction
#         model_path = os.path.join(settings.BASE_DIR, 'trained_model.pt')
#         model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=8)
#         model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#         model.eval()

#         transform = transforms.Compose([
#             transforms.Resize(299),
#             transforms.CenterCrop(299),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

#         image_path = os.path.join(media_dir, f'captured_image.jpg')
#         image = Image.open(image_path)
#         input_tensor = transform(image)
#         input_batch = input_tensor.unsqueeze(0)
        
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = model.to(device)
#         input_batch = input_batch.to(device)
        
#         with torch.no_grad():
#             output = model(input_batch)

#         predicted_class_index = torch.argmax(output, dim=1).item()

#         data_dir = os.path.join(settings.BASE_DIR, 'media', 'patient')
#         class_labels = sorted(os.listdir(data_dir))
#         predicted_class_label = class_labels[predicted_class_index]
#         print("Predicted Class Label:", predicted_class_label)

#         patient = get_object_or_404(Patient, name=predicted_class_label)
#         if patient.email == email:
#             patient_records = PatientRecord.objects.filter(name=predicted_class_label)
#             return render(request, 'app/patient/patient_cred_result.html', {'patient_records': patient_records})
#         else:
#             return JsonResponse({'result': 'Email does not match'}, status=400)

#     return render(request, 'app/patient/image_capture_patient.html')

def patient_capture_email(request):
    email = request.GET.get('email', '')
    if request.method == 'POST':
        frame = request.POST.get('image_data')
        img_data = frame.split(",")[1]
        img_data = bytes(img_data, 'utf-8')

        media_dir = os.path.join('media/capture')
        if not os.path.exists(media_dir):
            os.makedirs(media_dir)

        with open(os.path.join(media_dir, 'captured_image.jpg'), 'wb') as f:
            f.write(base64.b64decode(img_data))
            
        model_path = os.path.join(settings.BASE_DIR, 'trained_model.pt')
        model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=8)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_path = os.path.join(settings.MEDIA_ROOT, 'capture', 'captured_image.jpg')
        image = Image.open(image_path)
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_batch = input_batch.to(device)
        
        with torch.no_grad():
            output = model(input_batch)

        predicted_class_index = torch.argmax(output, dim=1).item()

        data_dir = os.path.join(settings.BASE_DIR, 'media', 'patient')
        class_labels = sorted(os.listdir(data_dir))
        predicted_class_label = class_labels[predicted_class_index]
        print("Predicted Class Label:", predicted_class_label)

        patient = get_object_or_404(Patient, name=predicted_class_label)
        if patient.email == email:
            patient_records = PatientRecord.objects.filter(name=predicted_class_label)
            return render(request, 'app/patient/patient_cred_result.html', {'patient_records': patient_records})
        else:
            return JsonResponse({'result': 'Email does not match'}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)
