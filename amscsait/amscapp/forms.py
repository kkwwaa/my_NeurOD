from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import get_user_model
from .models import Patient, Probs, Question, Option
from django.contrib.auth.models import User

User = get_user_model()

class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question
        fields = ['question_text', 'avg_value', 'min_value', 'max_value']

class OptionForm(forms.ModelForm):
    class Meta:
        model = Option
        fields = ['name', 'score']

class ProbeForm(forms.ModelForm):
    class Meta:
        model = Probs
        fields = ['title', 'modal']
        labels = {
            'title': 'Название',
            'modal': 'Модальность',
        }


class UserLoginForm(AuthenticationForm):
    username = forms.CharField(
        label="Имя пользователя",
        widget=forms.TextInput(attrs={"class": "form-control"}),
    )
    password = forms.CharField(
        label="Пароль", widget=forms.PasswordInput(attrs={"class": "form-control"})

    )


class UserRegisterForm(UserCreationForm):
    error_messages = {}

    email = forms.EmailField(
        label="Электронная почта",
        widget=forms.EmailInput(attrs={"class": "form-control"}),
        error_messages={}
    )
    first_name = forms.CharField(
        label="Имя",
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={"class": "form-control"}),
        error_messages={}
    )
    last_name = forms.CharField(
        label="Фамилия",
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={"class": "form-control"}),
        error_messages={}
    )
    password1 = forms.CharField(
        label="Пароль",
        widget=forms.PasswordInput(attrs={"class": "form-control"}),
        error_messages={}
    )
    password2 = forms.CharField(
        label="Подтверждение пароля",
        widget=forms.PasswordInput(attrs={"class": "form-control"}),
        error_messages={}
    )
    username = forms.CharField(
        label="Имя",
        widget=forms.TextInput(attrs={"class": "form-control"}),
        error_messages={}
    )

    class Meta:
        model = User
        fields = ["username", "email", "first_name", "last_name", "password1", "password2"]


class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ["name", "date_of_birth", "group", "gender", "parent_name", "parent_phone_number", "parent_email", "plaint", "goal", "diagnosis", "anamnes", 'exam_data', "hand"]
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-input'}),
            'plaint': forms.Textarea(attrs={'cols': 60, 'rows': 8}),
            'goal': forms.Textarea(attrs={'cols': 60, 'rows': 5}),
            'diagnosis': forms.Textarea(attrs={'cols': 60, 'rows': 6}),
            'anamnes': forms.Textarea(attrs={'cols': 60, 'rows': 8}),
            'exam_data': forms.Textarea(attrs={'cols': 60, 'rows': 8}),
            "date_of_birth": forms.DateInput(attrs={"class": "datepicker"}),
        }

