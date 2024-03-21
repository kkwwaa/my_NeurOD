import math, time
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .forms import (PatientForm, UserLoginForm,
                    UserRegisterForm, OptionForm)
from .models import Patient, PatientAnswer, Block, Probs, Modality, Question, PatientNumericAnswer, NumericQuestion, ProbsImage, NumericOption
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


def get_severity(score, mean, std_dev, rev=False):
    lower_bound = abs(mean - std_dev)
    upper_bound = mean + std_dev

    if rev:
        if score >= lower_bound:
            return "Выше среднего"
        elif score < math.ceil(upper_bound):
            return "Ниже среднего"
        else:
            return "Средне"
    else:
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
        for answer, severity,score_sum in result['severity_list']:
            if hasattr(answer, 'option'):
                row = {
                    'Проба': answer.proba,
                    'Результаты пробы': '<a href="/patient/{}/probs_results/{}">Просмотреть результаты</a>'.format(answer.patient.pk, answer.proba.slug),
                    'Вопрос': answer.question,
                    'Ответ': answer.option,
                    'Кол-во баллов': score_sum[0],
                    'Оценка тяжести': severity[0],
                }
            else:
                row = {
                    'Проба': answer.proba,
                    'Результаты пробы': '<a href="/patient/{}/probs_results/{}">Просмотреть результаты</a>'.format(answer.patient.pk, answer.proba.slug),
                    'Вопрос': answer.question,
                    'Ответ': answer.answer,
                    'Кол-во баллов': score_sum[0],
                    'Оценка тяжести': severity[0],
                }
            data.append(row)
        row = {
            'Проба': '',
            'Вопрос': 'Итого:',
            'Ответ': result['total_score'],
            'Кол-во баллов': result['score_sum'],
            'Оценка тяжести': f'{result["sum_score"]}%'
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.fillna('', inplace=True)
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
        score_sum = []

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

                score_sum.append((score_severity(severity[-1][0]), answer.created_at))
                answers.append(answer)
        max_num_dict = {}
        severity_num = []
        for answer in modality_numeric_answers:
            proba = answer.proba
            num = answer.num
            if proba not in max_num_dict or num > max_num_dict[proba]:
                max_num_dict[proba] = num

        for answer in modality_numeric_answers:
            if (answer.proba in max_num_dict) and answer.num == max_num_dict[answer.proba]:
                total_score += answer.answer
                question = answer.question
                avg_value = question.avg_value

                if avg_value is not None:
                    std_dev = abs(question.max_value - question.min_value) / 4
                    severity_num.append((get_severity(answer.answer, avg_value, std_dev, question.rev), answer.created_at))

                score_sum.append((score_severity(severity_num[-1][0]), answer.created_at))

                answers.append(answer)

        answers = sorted(answers, key=lambda x: x.created_at)
        severity = sorted(severity + severity_num, key=lambda x: x[1])
        score_sum = sorted(score_sum, key=lambda x: x[1])
        if len(severity) != 0:
            sum_score = round((sum(score[0] for score in score_sum) / len(severity)) * 100,2)
            #modality.develop = sum_score
            #modality.save()
        else:
            sum_score=''
        result = {
            "modality": modality,
            'pk': pk,
            "answers": answers,
            "total_score": total_score,
            "severity_list": [(answer, severity,score_sum) for answer, severity,score_sum in zip(answers, severity,score_sum)],
            'sum_score': sum_score,
            'score_sum': sum(score[0] for score in score_sum),
        }

        #ЗАКЛЮЧЕНИЕ
        #zacl=zacluchenie(patient,modalities)

        results.append(result)
        #modality_names.append(result['modality'])
        modality_names=['Программирование\nи контроль',"Регуляция\nактивности",'Серийной\nорганизации','Переработка\nкинестетической\nинформации', 'Переработка\nслуховой\nинформации','Переработка\nзрительной\nинформации','Переработки\nзрительно\n-пространственной\nинформации',]
        score_sums.append(result['sum_score'])
        score_sums=score_sums[:7]

        df = generate_dataframe([result])

        # Преобразование DataFrame в HTML
        html_table = df.to_html(index=False, classes="table", justify="center", escape=False)

        # Добавление таблицы в словарь по ключу модальности
        html_tables[modality] = html_table
    chart_image_path= r'/static/image/graph.png'
    if request.method == 'GET':
        if 'update-chart' in request.GET:
            # Построение графика
            x = np.arange(len(modality_names))
            y = np.array(score_sums)
            #plt.plot(x, y, marker='o')

            plt.figure(figsize=(10, 6))  # Увеличение размера графика
            plt.plot(x, score_sums, marker='o')

            # Настройка внешнего вида графика
            plt.ylabel('Оценка (%)')
            plt.xticks(x, modality_names, rotation=0, wrap=True)
            plt.yticks(np.arange(0, 101, 10))
            chart_image_path = r'static/image/graph.png'
            plt.savefig(chart_image_path)

        elif 'clear-chart' in request.GET:
            # Очистка графика
            plt.clf()
            plt.plot([], [])
            plt.ylabel('Оценка (%)')
            plt.xticks([], [])
            plt.ylim(-2, 110)
            chart_image_path = r'static/image/graph.png'
            plt.savefig(chart_image_path)
    timestamp = int(time.time())
    zacl=test_zacl(pk)
    return render(
        request,
        "amscapp/view_patient.html",
        {"patient": patient, 'pk': pk,'timestamp': timestamp, "html_tables": html_tables, "results": results, 'chart_image_path': chart_image_path,
         "title": "Анкета пациента",'zacl': zacl},
    )


def test_zacl(pk):
    parts_zacl=[]
    modalities=['Анализ состояния функций поддержания тонуса и уровня бодроствования', 'Исследование функций серийной организации, программирования и контроля деятельности:','Исследование функций приёма, хранения и передачи информации:', 'Обработка кинестетической информации.', 'Обработка слуховой информации.','Обработка зрительной информации.','Обработка зрительно-пространственной информации.']
    #1 блок
    zacl=''
    modality=Modality.objects.get(name="I блока мозга")
    questions = modality.question_set.all()
    score=0
    for question in questions:
        answers = PatientAnswer.objects.filter(patient_id=pk, question_id=question.pk).order_by('-num')
        if answers.exists():
            answer = answers.first()
            if question.question_text=='Темп:':
                if answer.option=='замедленный':
                    zacl+=(' Снижен темп деятельности.')
                    score-=1
                else:
                    zacl+=(' Темп деятельности нормальный, иногда повышен - соответствует возрасту.')
            if question.question_text=='Тонус:':
                if answer.option=='снижение':
                    score-=1
                    zacl+=(' Пониженный тонус - общая слабость и утомляемость, расслабленность, снижение продуктивности и низкая концентрация внимания.')
                elif answer.option=='повышение':
                    score+=1
                    zacl+=(' Повышенный тонус при выполнении двигательных заданий, высокая напряженность, случаи синкинезии. Концентрация внимания на заданиях ребенку доступна.')
                else:
                    zacl+=(' Тонус стабильный без выраженных перепадов.')
            if question.question_text=='Импульсивность:':
                if answer.option=='нет':
                    zacl+=(' Отсутствие импульсивности, контроль поведения, обдумывание выбора.')
                else:
                    score+=1
                    if 'импульсив' not in zacl: zacl+=' Наличие импульсивных ответов, случаи отсутствия обдумывания заданий, скачки уровня внимания.'
    if score>1:
        zacl='выявляет заметные признаки избытка данных процессов.'+zacl
    elif score<-1:
        zacl = 'выявляет выраженные признаки дефицита данных процессов.'+zacl
    else:
        zacl = 'не выявляет выраженные признаки дефицита или избытка данных процессов.' + zacl
    parts_zacl.append(zacl)

    #Сер. орг. и программирование и контроль
    zacl=''
    proba=Probs.objects.get(title='Динамический праксис')
    questions=Question.objects.filter(proba_id=proba.pk).order_by('num')
    for question in questions:
        answers = PatientAnswer.objects.filter(patient_id=pk, question_id=question.pk).order_by('-num')
        if answers.exists():
            answer = answers.first()
            if question.question_text=='Выполнение:':
                if answer.option.score==0:
                    zacl+=' Проба на динамический праксис проходит от замедленного или пачками к плавному выполнению'
                elif answer.option.score==1:
                    zacl+=' Проба на динамический праксис проходит от поэлементного к плавному выполнению'
                elif answer.option.score==2:
                    zacl+=' Проба на динамический праксис проходит пачками сразу или после сбоев'
                else:
                    zacl += ' Проба на динамический праксис показывает поэлементное неавтоматическое выполнение'
            if question.question_text == 'Ошибки серийной организации:':
                if answer.option.score==0:
                    zacl+=', выполняется без ошибок.'
                elif answer.option.score==1:
                    zacl+=', в процессе допускает единичные ошибки'
                elif answer.option.score==2:
                    zacl+=', с повторяющимися сбоями и с тенденцией к расширению.'
                elif answer.option.score==3:
                    zacl += ', выявляються тенденции к расширению или сужению программы.'
                else:
                    zacl += ', инертный стереотип.'
            if question.question_text=='Усвоение двигательной программы:':
                if answer.option.score==0:
                    zacl+='Усвоение инструкций после первого предъявления.'
                elif answer.option.score==1:
                    zacl+='Затрудненное усвоение инструкций, после второго предъявления.'
                elif answer.option.score==2:
                    zacl+='Усвоение инструкций только после совместного выполнения.'
                elif answer.option.score==3:
                    zacl += 'Усвоение инструкций только после совместного выполнения  по речевой инструкции.'
                else:
                    zacl+="Неусвоение, уход и потеря программы."
    proba = Probs.objects.get(title='Реципрокная координация')
    questions = Question.objects.filter(proba_id=proba.pk).order_by('num')
    for question in questions:
        answers = PatientAnswer.objects.filter(patient_id=pk, question_id=question.pk).order_by('-num')
        if answers.exists():
            answer = answers.first()
            if question.question_text == 'Выполнение:':
                if answer.option.score == 0:
                    zacl += ' Проба на реципрокную координацию проходит плавно и автоматизировано, единичные сбои в начале. '
                elif answer.option.score == 1:
                    zacl += ' Проба на реципрокную координацию проходит нормально: переход к автоматизированным движениям после нескольких сбоев.'
                elif answer.option.score == 2:
                    zacl += ' Проба на реципрокную координацию проходит затрудненно: повторяющиеся сбои, отставания одной руки с самокоррекцией.'
                elif answer.option.score == 3:
                    zacl += ' Проба на реципрокную координацию проходит с вырженными затруднениями: поочередное выполнение.'
                else:
                    zacl += ' Проба на реципрокную координацию проходит тяжело с явными трудностями: уподобление движений обоих рук.'
    parts_zacl.append(zacl)

    #Прием, хранение и переработка инфы
    zacl = ''
    parts_zacl.append(zacl)
    questions_names = []
    options = []
    score = 0
    proba = Probs.objects.get(title='Праксис позы пальцев')
    questions = NumericQuestion.objects.filter(proba_id=proba.pk).order_by('num')
    zacl += ' При выполнении пробы на праксис позы пальцев по'
    for question in questions:
        answers = PatientNumericAnswer.objects.filter(patient_id=pk, question_id=question.pk).order_by('-num')
        if answers.exists():
            answer = answers.first()
            if question.question_text == 'Продуктивность по зрит. образу (правая):':
                zacl+=f' зрительному образцу делает на правой руке {answer.answer} ошибки '
            if question.question_text == 'Продуктивность по зрит. образу (левая):':
                zacl+=f'и {answer.answer} на левой руке. '
            if question.question_text == 'Продуктивность по проприоц. с переносом позы:':
                zacl += f', с переносом позы допускает {answer.answer} ошибки. '
            if question.num == 3:
                zacl += f' При выполнении пробы по проприоцептивному образцу без переноса делает {answer.answer} ошибки'
            if question.num==5:
                zacl+=f'Также допускает на правой руке {answer.answer} кинестетические ошибки, '
            if question.num == 6:
                zacl += f'в левой - {answer.answer}.'
            if question.num == 7:
                zacl += f' Были допущены {answer.answer} импульсивные ошибки.\n'
    parts_zacl.append(zacl)

    #Слух инфа
    zacl=''
    proba = Probs.objects.get(title='Слухоречевая память')
    questions = NumericQuestion.objects.filter(proba_id=proba.pk).order_by('num')
    for question in questions:
        answers = PatientNumericAnswer.objects.filter(patient_id=pk, question_id=question.pk).order_by('-num')
        if answers.exists():
            answer = answers.first()
            if question.num == 12:
                if answer.answer<1:
                    zacl += ' Вербальные ошибки и проблемы подбора слов встречаются редко; в целом словарь ребенка хорошо развит и находится в верхней части нормы для своего возраста.'
                elif answer.answer<3:
                    zacl += ' Вербальные ошибки и проблемы подбора слов встречаются; в целом словарь ребенка находится в рамках среднего для своего возраста.'
                else:
                    zacl += ' Вербальные ошибки и проблемы подбора слов встречаются часто; в целом словарь ребенка плохо развит и находится ниже нормативов для своего возраста.'
            #числа в закл
            if question.num==2:
                a=answer.answer
            if question.num == 4:
                b = answer.answer
            if question.num == 6:
                c = answer.answer
                zacl += f' В пробе на запоминание 2 групп по 3 слова кривая заучивания имеет вид {a}-{b}-{c}. Продуктивность повторений распределяется по следующиму графику {a1}-{b1}-{c1}.'
            if question.num==1:
                a1=answer.answer
            if question.num == 3:
                b1 = answer.answer
            if question.num == 5:
                c1 = answer.answer
    parts_zacl.append(zacl)

    #Зрит инфа
    zacl=''
    proba = Probs.objects.get(title='Зрительный гнозис наложенные изображения')
    questions = NumericQuestion.objects.filter(proba_id=proba.pk).order_by('num')
    for question in questions:
        answers = PatientNumericAnswer.objects.filter(patient_id=pk, question_id=question.pk).order_by('-num')
        if answers.exists():
            answer = answers.first()
            if question.num == 1:
                zacl += f' В пробах на зрительный гнозис опознаны {answer.answer} из {int(question.max_value)} наложенных изображений, '
    proba = Probs.objects.get(title='Зрительный гнозис перечеркнутые изображения')
    questions = NumericQuestion.objects.filter(proba_id=proba.pk).order_by('num')
    for question in questions:
        answers = PatientNumericAnswer.objects.filter(patient_id=pk, question_id=question.pk).order_by('-num')
        if answers.exists():
            answer = answers.first()
            if question.num == 1:
                zacl += f'{answer.answer} из {int(question.max_value)} перечеркнутых изображений и '
    proba = Probs.objects.get(title='Зрительный гнозис недорисованные изображения')
    questions = NumericQuestion.objects.filter(proba_id=proba.pk).order_by('num')
    for question in questions:
        answers = PatientNumericAnswer.objects.filter(patient_id=pk, question_id=question.pk).order_by('-num')
        if answers.exists():
            answer = answers.first()
            if question.num == 1:
                zacl += f'{answer.answer} из {int(question.max_value)} недорисованных изображений. '
    parts_zacl.append(zacl)

    #зрит-простр инфа
    zacl = ''
    proba = Probs.objects.get(title='Проба Хеда')
    question = NumericQuestion.objects.get(proba_id=proba.pk, num=2)
    answers = PatientNumericAnswer.objects.filter(patient_id=pk, question_id=question.pk).order_by('-num')
    if answers.exists():
        answer = answers.first()
        if answer.answer<3:
            zacl+=' Пространственные ошибки в моторных пробах встречаются редко.'
        else:
            zacl+=' Распространены пространственные ошибки в моторных пробах.'

    proba = Probs.objects.get(title='Копирование домика')
    questions = Question.objects.filter(proba_id=proba.pk).order_by('num')
    for question in questions:
        answers = PatientAnswer.objects.filter(patient_id=pk, question_id=question.pk).order_by('-num')
        if answers.exists():
            answer = answers.first()
            if question.num == 1:
                zacl += f' При копировании трехмерного изображения дома наблюдается со стороны левополушарной стратегии: {answer.option}.'
            if question.num==2:
                zacl += f' Также со стороны правополушарной стратегии заметна следующая тенденция: {answer.option}.'
        else: options.append('No')
    parts_zacl.append(zacl)

    return(zip(parts_zacl, modalities))


@login_required
def probs_results(request, pk, slug):
    patient = get_object_or_404(Patient, pk=pk)
    proba = Probs.objects.filter(slug=slug).first()
    results = []

    if request.method == 'POST':
        prob_to_delete = request.POST.get('prob_to_delete')
        num = request.POST.get('num')
        if prob_to_delete:
            PatientAnswer.objects.filter(patient_id=pk, proba=prob_to_delete, num=num).delete()
            PatientNumericAnswer.objects.filter(patient_id=pk, proba=prob_to_delete, num=num).delete()

    num1 = patient.patientanswer_set.filter(proba=proba.pk).values_list('num', flat=True).distinct()
    if num1:
        nums=num1
    else:
        nums = patient.patientnumericanswer_set.filter(proba=proba.pk).values_list('num', flat=True).distinct()
    for num in nums:
        #proba_answers = patient.patientanswer_set.filter(proba=proba, num=num)
        proba_answers = PatientAnswer.objects.filter(proba=proba, num=num, patient=patient)
        proba_answers = proba_answers.distinct('question', 'option')  # ВОЗМОЖНА ОШИБКА, выводит уникальные по вопросу и опции
        total_score = sum(answer.option.score for answer in proba_answers)
        total_score = total_score if total_score is not None else 0
        numeric_answers = PatientNumericAnswer.objects.filter(proba=proba, num=num, patient=patient)
        numeric_answers = numeric_answers.distinct('question','question__options__name') # ВОЗМОЖНА ОШИБКА
        numeric_total_score = sum(answer.answer for answer in numeric_answers)
        numeric_total_score = numeric_total_score if numeric_total_score is not None else 0
        proba1 = Probs.objects.get(pk=proba.pk)

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
        {"results": results, 'nums': nums, 'proba': proba, 'pk': pk},
    )


@login_required
def probs_list(request, pk):
    modalities = Block.objects.all()
    probs = Probs.objects.all()

    # Определите переменную для поискового запроса
    search_query = request.GET.get('search', '')

    # Фильтруйте и сортируйте пробы в зависимости от наличия поискового запроса и значения sort
    sort = request.GET.get('sort', 'title')  # По умолчанию сортировка по полю title

    if search_query:
        probs = probs.filter(title__icontains=search_query)

    # if sort == 'numer':
    #     probs = probs.order_by('numer')
    # else:
    probs = probs.order_by('title')


    return render(request, 'amscapp/probs_list.html', {'modalities': modalities, 'probs': probs, 'pk': pk, 'search_query': search_query})


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
        num1 = int((PatientNumericAnswer.objects.filter(patient_id=pk, proba=proba).count()) // (
            NumericQuestion.objects.filter(proba=proba).count()))

    if request.method == 'POST':

        for question in questions:
            answer_option_id = request.POST.get(f'question_{question.id}', None)
            if answer_option_id is not None:
                for modal in question.modality.all():

                    PatientAnswer.objects.create(
                        patient_id=pk,
                        question=question,
                        option_id=answer_option_id,
                        proba=proba,
                        modal=modal,
                        num=num,
                    )

        for numeric_question in numeric_questions:
            total_sum = 0
            for option in numeric_question.options.all():
                numeric_answer_value = request.POST.get(f'numeric_question_{numeric_question.id}_{option.id}', None)
                if numeric_answer_value is not None:
                    for modal in numeric_question.modality.all():
                        total_sum = int(numeric_answer_value) * option.coefficient

                        PatientNumericAnswer.objects.create(
                            patient_id=pk,
                            question=numeric_question,
                            answer=total_sum,
                            proba=proba,
                            modal=modal,  # Здесь modal - объект Modality
                            num=num1,
                        )

        return redirect('probs_list', pk=pk)

    try:
        proba_image = ProbsImage.objects.get(prob=proba)
    except ProbsImage.DoesNotExist:
        proba_image = None

    context = {
        'prob': proba,
        'proba_image': proba_image,
        'questions': sorted(questions, key=lambda q: (q.num if q.num is not None else float('inf'))),
        'numeric_questions': sorted(numeric_questions, key=lambda q: (q.num if q.num is not None else float('inf'))),
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


#@user_passes_test(lambda u: not u.is_authenticated)
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
