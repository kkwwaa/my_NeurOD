import math, time
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .forms import (PatientForm, UserLoginForm,
                    UserRegisterForm, OptionForm)
from .models import Patient, PatientAnswer, Block, Probs, Modality, Question, PatientNumericAnswer, NumericQuestion
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required, user_passes_test, permission_required
from django.db.models import Q, Sum, Min, Max, Avg
from django.shortcuts import get_object_or_404, redirect, render
from django.http import HttpResponse, HttpResponseNotFound, Http404
from django.forms import formset_factory

OptionFormSet = formset_factory(OptionForm, extra=1)


@login_required
def index(request):
    user = request.user
    query = request.GET.get('q')

    if query:
        patients = Patient.objects.filter(user=user).filter(
            Q(name__icontains=query) | Q(group__icontains=query)
        )
    else:
        patients = Patient.objects.filter(user=user).order_by("name")

    return render(request, "amscapp/index.html", {"patients": patients})


@login_required
def create_patient(request):
    form = PatientForm(request.POST or None)
    if form.is_valid():
        patient = form.save(commit=False)
        patient.user = request.user
        patient.save()
        messages.success(request, "Пациент добавлен")
        return redirect("view_patient", pk=form.instance.pk)
    return render(
        request,
        "amscapp/post_form.html",
        {"form": form, "title": "Добавление пациента", "submit_text": "Добавить пациента"},
    )


def get_severity(score, mean, std_dev):
    lower_bound = abs(mean - std_dev)
    upper_bound = mean + std_dev

    if score <= lower_bound:
        return "Выше среднего"
    elif score > math.ceil(upper_bound):
        return "Ниже среднего"
    else:
        return "Средне"


def score_severity(severity_num):
    if severity_num == "Выше среднего":
        return 1
    elif severity_num == "Средне":
        return 0.5
    elif severity_num == "Ниже среднего":
        return 0.2


def generate_dataframe(results):
    data = []
    for result in results:
        for answer, severity in result['severity_list']:
            if hasattr(answer, 'option'):
                row = {
                    'Проба': answer.proba,
                    'Вопрос': answer.question,
                    'Ответ': answer.option,
                    'Кол-во баллов': answer.option.score,
                    'Оценка тяжести': severity[0],
                }
            else:
                row = {
                    'Проба': answer.proba,
                    'Вопрос': answer.question,
                    'Ответ': answer.answer,
                    'Кол-во баллов': answer.answer,
                    'Оценка тяжести': severity[0],
                }
            data.append(row)
        row = {
            'Проба': '',
            'Вопрос': '',
            'Ответ': 'Итого:',
            'Кол-во баллов': result['total_score'],
            'Оценка тяжести': f'{result["score_sum"]}%'
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.loc[df['Проба'].duplicated(), 'Проба'] = ''
    return df


@login_required
def view_patient(request, pk):
    patient = get_object_or_404(Patient, pk=pk, user=request.user)
    modalities = Modality.objects.all()
    questions = Question.objects.filter(modality__isnull=False).distinct()
    num_questions = NumericQuestion.objects.filter(modality__isnull=False).distinct()
    results = []
    modality_names = []
    score_sums = []
    html_tables = {}

    for modality in modalities:
        modality_answers = PatientAnswer.objects.filter(patient=patient, modal=modality)
        modality_numeric_answers = PatientNumericAnswer.objects.filter(patient=patient, modal=modality)
        total_score = 0
        severity = []
        answers = []
        score_sum = 0

        max_num_dict = {}
        for answer in modality_answers:
            proba = answer.proba
            num = answer.num
            if proba not in max_num_dict or num > max_num_dict[proba]:
                max_num_dict[proba] = num

        for answer in modality_answers:
            if answer.proba in max_num_dict and answer.num == max_num_dict[answer.proba]:
                total_score += answer.option.score
                question = answer.question
                avg_value = question.avg_value

                if avg_value is not None:
                    std_dev = abs(question.max_value - question.min_value) / 4
                    severity.append((get_severity(answer.option.score, avg_value, std_dev), answer.created_at))

                score_sum += score_severity(severity[-1][0])
                answers.append(answer)

        severity_num = []
        for answer in modality_numeric_answers:
            if answer.proba in max_num_dict and answer.num == max_num_dict[answer.proba]:
                total_score += answer.answer
                question = answer.question
                avg_value = question.avg_value

                if avg_value is not None:
                    std_dev = abs(question.max_value - question.min_value) / 4
                    severity_num.append((get_severity(answer.answer, avg_value, std_dev), answer.created_at))

                score_sum += score_severity(severity_num[-1][0])
                answers.append(answer)

        if (questions.filter(modality=modality).count() + num_questions.filter(modality=modality).count()) != 0:
            score_sum = (score_sum / (questions.filter(modality=modality).count() + num_questions.filter(
                modality=modality).count())) * 100

        answers = sorted(answers, key=lambda x: x.created_at)
        severity = sorted(severity + severity_num, key=lambda x: x[1])

        result = {
            "modality": modality,
            "answers": answers,
            "total_score": total_score,
            "severity_list": [(answer, severity) for answer, severity in zip(answers, severity)],
            'score_sum': round(score_sum, 2),
        }
        results.append(result)
        modality_names.append(result['modality'])
        score_sums.append(result['score_sum'])

        df = generate_dataframe([result])

        # Преобразование DataFrame в HTML
        html_table = df.to_html(index=False, classes="table")

        # Добавление таблицы в словарь по ключу модальности
        html_tables[modality] = html_table

    if request.method == 'GET':
        if 'update-chart' in request.GET:
            # Построение графика
            x = np.arange(len(modality_names))
            y = np.array(score_sums)
            plt.plot(x, y, marker='o')

            # Настройка внешнего вида графика
            plt.xlabel('Модальности')
            plt.ylabel('Оценка (%)')
            plt.title('График развития модальностей')
            plt.xticks(x, modality_names, rotation=90)
            plt.ylim(-2, 102)

            # Сохранение изображения графика
            chart_image_path = r'static/image/graph.png'
            plt.savefig(chart_image_path)

        elif 'clear-chart' in request.GET:
            # Очистка графика
            plt.clf()
            plt.plot([], [])

            # Настройка внешнего вида графика
            plt.xlabel('Модальности')
            plt.ylabel('Оценка (%)')
            plt.title('График развития модальностей')
            plt.xticks([], [])
            plt.ylim(-2, 110)
            chart_image_path = r'static/image/graph.png'
            plt.savefig(chart_image_path)
    timestamp = int(time.time())
    return render(
        request,
        "amscapp/view_patient.html",
        {"patient": patient, 'pk': pk, 'timestamp': timestamp, "html_tables": html_tables, "results": results,
         "title": "Анкета пациента"},
    )


@login_required
def probs_results(request, pk, modal):
    patient = get_object_or_404(Patient, pk=pk)
    probs = patient.patientanswer_set.filter(modal_id=modal).values_list('proba', flat=True).distinct()
    results = []

    if request.method == 'POST':
        prob_to_delete = request.POST.get('prob_to_delete')
        num = request.POST.get('num')
        if prob_to_delete:
            PatientAnswer.objects.filter(patient_id=pk, proba=prob_to_delete, num=num).delete()
            PatientNumericAnswer.objects.filter(patient_id=pk, proba=prob_to_delete, num=num).delete()

    for proba in probs:
        nums = patient.patientanswer_set.filter(proba=proba).values_list('num', flat=True).distinct()
        for num in nums:
            proba_answers = patient.patientanswer_set.filter(proba=proba, num=num)
            total_score = proba_answers.aggregate(total_score=Sum("option__score"))["total_score"]
            total_score = total_score if total_score is not None else 0
            numeric_answers = patient.patientnumericanswer_set.filter(proba=proba, num=num)
            numeric_total_score = numeric_answers.aggregate(total_score=Sum("answer"))["total_score"]
            numeric_total_score = numeric_total_score if numeric_total_score is not None else 0
            proba1 = Probs.objects.get(pk=proba)

            result = {
                "num": num,
                "proba": proba1,
                'proba_pk': proba,
                "answers": proba_answers,
                "numeric_answers": numeric_answers,
                "total_score": total_score + numeric_total_score,
            }
            results.append(result)

    return render(
        request,
        "amscapp/probs_results.html",
        {"results": results},
    )


@login_required
def probs_list(request, pk):
    modalities = Block.objects.all()
    probs = Probs.objects.all()
    return render(request, 'amscapp/probs_list.html', {'modalities': modalities, 'probs': probs, 'pk': pk})


@login_required
def proba(request, pk, proba_pk):
    # Получение объекта пробы по ее ID
    try:
        proba = Probs.objects.get(pk=proba_pk)
    except Probs.DoesNotExist:
        return HttpResponse("Ошибка: Проба не найдена.")

    # Получение всех вопросов, связанных с пробой
    questions = Question.objects.filter(proba=proba).distinct().order_by('question_text')
    numeric_questions = NumericQuestion.objects.filter(proba=proba).distinct().order_by('question_text')
    if Question.objects.filter(proba=proba).count() != 0:
        num = (PatientAnswer.objects.filter(patient_id=pk, proba=proba).count()) // (
            Question.objects.filter(proba=proba).count())
    if NumericQuestion.objects.filter(proba=proba).count() != 0:
        num1 = (PatientNumericAnswer.objects.filter(patient_id=pk, proba=proba).count()) // (
            NumericQuestion.objects.filter(proba=proba).count())

    if request.method == 'POST':

        # Обработка отправленных ответов на вопросы
        for question in questions:
            answer_option_id = request.POST.get(f'question_{question.id}', None)
            if answer_option_id is not None:
                # Получение модальности из связи вопрос-модальность
                modal = question.modality

                # Создание объекта PatientAnswer и сохранение в БД с указанием модальности или None
                PatientAnswer.objects.create(
                    patient_id=pk,
                    question=question,
                    option_id=answer_option_id,
                    proba=proba,
                    modal=modal,
                    num=num,
                )

        for numeric_question in numeric_questions:
            total_sum = 0  # Переменная для хранения суммы ответов на числовой вопрос
            for option in numeric_question.options.all():
                numeric_answer_value = request.POST.get(f'numeric_question_{numeric_question.id}_{option.id}', None)
                if numeric_answer_value is not None:
                    # Получение модальности из связи вопрос-модальность
                    modal = numeric_question.modality
                    total_sum += int(numeric_answer_value) * option.coefficient

                    # Создание объекта PatientNumericAnswer и сохранение в БД с указанием модальности или None
            PatientNumericAnswer.objects.create(
                patient_id=pk,
                question=numeric_question,
                answer=total_sum,
                proba=proba,
                modal=modal,
                num=num1,
            )

        return redirect('probs_list', pk=pk)  # Перенаправление на страницу успешного сохранения ответов

    context = {
        'prob': proba,
        'questions': questions,
        'numeric_questions': numeric_questions,
    }

    return render(request, 'amscapp/proba_detail.html', context)


@login_required
def edit_patient(request, pk):
    patient = get_object_or_404(Patient, pk=pk)
    form = PatientForm(request.POST or None, instance=patient)
    if form.is_valid():
        form.save()
        messages.success(request, "Пациент изменен")
        return redirect("view_patient", pk=form.instance.pk)
    return render(
        request, "amscapp/post_form.html",
        {"form": form, "title": "Изменение пациента", "submit_text": "Применить изменения"}
    )


@user_passes_test(lambda u: not u.is_authenticated)
def register(request):
    form = UserRegisterForm(request.POST or None)
    if form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, "Регистрация прошла успешно!")
        return redirect("index")
    return render(
        request, "amscapp/post_form.html", {"form": form, "title": "Регистрация", "submit_text": "Регистрация"}
    )


@user_passes_test(lambda u: not u.is_authenticated)
def user_login(request):
    form = UserLoginForm(request, request.POST or None)
    if form.is_valid():
        user = form.get_user()
        login(request, user)
        messages.success(request, "Вы успешно вошли!")
        return redirect("index")
    return render(request, "amscapp/post_form.html", {"form": form, "title": "Авторизация", "submit_text": "Вход"})


@login_required
def exit(request):
    logout(request)
    return redirect("login")


def create_proba(request):
    # QuestionFormSet = formset_factory(QuestionForm)
    #
    # if request.method == 'POST':
    #     form = ProbeForm(request.POST)
    #     formset = QuestionFormSet(request.POST)
    #     if form.is_valid() and formset.is_valid():
    #         proba = form.save()
    #         for question_form in formset:
    #             question = question_form.save(commit=False)
    #             question.proba = proba
    #             question.modality = proba.modal
    #             question.save()
    #         return redirect('index')  # Перенаправление на страницу с деталями пробы
    # else:
    #     form = ProbeForm()
    #     formset = QuestionFormSet()

    return HttpResponse("Пффффффффф...")


def admin(request):
    return render(request, 'admin/')
