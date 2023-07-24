from datetime import date
from enum import Enum
from django.urls import reverse
from django.core import validators
from django.utils.html import mark_safe
from django.core.validators import MinValueValidator
from django.db.models.signals import pre_save
from django.dispatch import receiver
from django.db import models
from django.contrib.auth.models import User
from django.utils.text import slugify
from unidecode import unidecode

class QuestionType(Enum):
    POLL = 'poll'
    TEXT = 'text'


class Block(models.Model):
    name = models.CharField("Блок",max_length=100)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Блок"
        verbose_name_plural = "Блоки"

class Modality(models.Model):
    name = models.CharField(max_length=100)
    block = models.ForeignKey(Block, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Модальность"
        verbose_name_plural = "Модальности"

class Question(models.Model):
    type_ = QuestionType.POLL
    num = models.IntegerField("Номер вопроса", null=True, blank=True)
    question_text = models.CharField("Текст вопроса", max_length=100)
    proba = models.ForeignKey(to="Probs", on_delete=models.CASCADE)
    modality = models.ManyToManyField(Modality)
    avg_value = models.FloatField("Среднее значение", null=True, blank=True)
    min_value = models.FloatField("Минимальное значение", null=True, blank=True)
    max_value = models.FloatField("Максимальное значение", null=True, blank=True)

    def __str__(self):
        return self.question_text

    class Meta:
        verbose_name = "Качественный вопрос"
        verbose_name_plural = "Качественные вопросы"


class Option(models.Model):
    name = models.CharField("Вариант ответа", max_length=100)
    score = models.FloatField("Колличество баллов", default=0)
    question = models.ForeignKey(
        "Question", related_name="options", on_delete=models.CASCADE
    )

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Вариант ответа"
        verbose_name_plural = "Варианты ответа"

class NumericQuestion(models.Model):
    question_text = models.CharField("Текст вопроса", max_length=100)
    num = models.IntegerField("Номер вопроса", null=True, blank=True)
    proba = models.ForeignKey(to="Probs", on_delete=models.CASCADE)
    modality = models.ManyToManyField(Modality)
    avg_value = models.FloatField("Среднее значение", null=True, blank=True)
    min_value = models.FloatField("Минимальное значение", null=True, blank=True)
    max_value = models.FloatField("Максимальное значение", null=True, blank=True)
    rev = models.BooleanField(default=False)
    def __str__(self):
        return self.question_text

    class Meta:
        verbose_name = "Числовый вопрос"
        verbose_name_plural = "Числовые вопросы"

class NumericOption(models.Model):
    name = models.CharField("Вариант ответа", max_length=100)
    coefficient = models.FloatField("Коэффициент", default=1)
    question = models.ForeignKey(
        "NumericQuestion", related_name="options", on_delete=models.CASCADE
    )

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Числовая опция"
        verbose_name_plural = "Числовые опции"

class Probs(models.Model):
    title = models.CharField(max_length=255)
    modal = models.ForeignKey(Modality, on_delete=models.CASCADE)
    block = models.ForeignKey(Block, on_delete=models.CASCADE, blank=True, null=True)
    slug = models.SlugField(unique=True, blank=True)
    numer = models.IntegerField("Номер в протоколе",null=True)

    def get_results_url(self):
        return reverse('probs_results', kwargs={'pk': self.pk, 'slug': self.slug})

    def save(self, *args, **kwargs):
        # Преобразуем название пробы в английский алфавит и создаем слаг
        if not self.slug:
            english_title = unidecode(self.title)  # Преобразуем название в английский
            self.slug = slugify(english_title)
        super(Probs, self).save(*args, **kwargs)

    def __str__(self):
        return self.title

    class Meta:
        verbose_name = "Проба"
        verbose_name_plural = "Пробы"


@receiver(pre_save, sender=Probs)
def set_block_from_modal(sender, instance, **kwargs):
    if not instance.block:  # Проверяем, если поле block не заполнено
        instance.block = instance.modal.block

class ProbsImage(models.Model):
    prob = models.ForeignKey(Probs, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='static/image/probs_img/')

    def image_tag(self):
        return mark_safe(f'<img src="{self.image.url}" alt="{self.prob.title}" style="max-width: 100%; height: auto;">')
    image_tag.short_description = 'Image'

    class Meta:
        verbose_name = "Инструкция к пробе"
        verbose_name_plural = "Инструкции к пробам"

class PatientAnswer(models.Model):
    patient = models.ForeignKey(to="Patient", on_delete=models.CASCADE)
    question = models.ForeignKey(to="Question", on_delete=models.CASCADE)
    option = models.ForeignKey(to="Option", on_delete=models.CASCADE)
    proba = models.ForeignKey(to="Probs", on_delete=models.CASCADE)
    modal = models.ForeignKey(to="Modality", on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    num = models.IntegerField("Повтор",default=0)
    class Meta:
        verbose_name = "Ответ анкеты"
        verbose_name_plural = "Ответы анкеты"
        db_table = "amscapp_patientanswer"

class PatientNumericAnswer(models.Model):
    patient = models.ForeignKey(to="Patient", on_delete=models.CASCADE)
    question = models.ForeignKey("NumericQuestion", on_delete=models.CASCADE)
    answer = models.IntegerField("Ответ", validators=[MinValueValidator(0)], blank=True)
    proba = models.ForeignKey(to="Probs", on_delete=models.CASCADE)
    modal = models.ForeignKey(to="Modality", on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    num = models.IntegerField("Повтор", default=0)
    class Meta:
        verbose_name = "Ответ на числовый вопрос"
        verbose_name_plural = "Ответы на числовые вопросы"
        db_table = "amscapp_patientnumericanswer"


class Patient(models.Model):
    class GenderChoice(models.TextChoices):
        MAN = "Мужской", "Мужской"
        WOMAN = "Женский", "Женский"

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField("ФИО", max_length=100)
    date_of_birth = models.DateField("Дата рождения")
    group = models.IntegerField(validators=[validators.MinValueValidator(1), validators.MaxValueValidator(120)],
                                verbose_name='Класс(группа)')
    gender = models.CharField(
        "Пол пациента", max_length=7, choices=GenderChoice.choices, default=GenderChoice.MAN
    )
    #address = models.CharField("Домашний адрес", max_length=100, blank=True)
    parent_name = models.CharField("ФИО родителя/опекуна", max_length=100, blank=True)
    parent_phone_number = models.CharField(
        "Номер контактного телефона родителя/опекуна", max_length=100, blank=True
    )
    parent_email = models.EmailField(
        "Электронная почта родителя/опекуна", max_length=100, blank=True
    )
    plaint = models.TextField(blank=True, verbose_name='Жалобы')
    goal = models.TextField(blank=True, verbose_name='Цель обращения')
    diagnosis = models.TextField(blank=True, verbose_name='Диагноз')
    anamnes = models.TextField(blank=True, verbose_name='Анамнез')
    exam_data = models.TextField(blank=True, verbose_name='Данные наблюдения')

    LEFT_HAND = 'Л'
    RIGHT_HAND = 'П'
    HAND_CHOICES = [(LEFT_HAND, 'Левая'), (RIGHT_HAND, 'Правая'), ]
    hand = models.CharField(max_length=1, choices=HAND_CHOICES, default=RIGHT_HAND, verbose_name='Ведущая рука')

    date_registration = models.DateTimeField("Дата регистрации", auto_now_add=True)

    def __str__(self):
        return self.name

    def calculate_age(self):
        today = date.today()
        return today.year - self.date_of_birth.year - (
                (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day))

    class Meta:
        verbose_name = "Пациент"
        verbose_name_plural = "Пациенты"
        db_table = "amscapp_patient"