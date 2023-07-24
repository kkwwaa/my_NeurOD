from django.contrib import admin
from django.contrib.auth.models import User
from import_export.widgets import ForeignKeyWidget

from .models import Option, Block, Modality, PatientNumericAnswer, PatientAnswer, Patient, Question, Probs, \
    NumericQuestion, NumericOption, ProbsImage
from import_export import resources, fields
from import_export.admin import ImportExportModelAdmin

class PatientAnswerInLineAdmin(admin.TabularInline):
    model = PatientAnswer
    extra = 0


class PatientResource(resources.ModelResource):
    class Meta:
        model = Patient

class PatientNumericAnswerInline(admin.TabularInline):
    model = PatientNumericAnswer
    extra = 0

@admin.register(Patient)
class PatientAdmin(ImportExportModelAdmin):
    inlines = [PatientAnswerInLineAdmin, PatientNumericAnswerInline]
    resource_class = PatientResource

class PatientAnswerResource(resources.ModelResource):
    id = fields.Field(column_name='id', attribute='id')
    patient = fields.Field(column_name='Пациент', attribute='patient', widget=ForeignKeyWidget(Patient, 'name'))
    question = fields.Field(column_name='Вопрос', attribute='question',
                            widget=ForeignKeyWidget(Question, 'question_text'))
    option = fields.Field(column_name='Ответ', attribute='option',
                          widget=ForeignKeyWidget(Option, 'name'))
    doctor = fields.Field(column_name='Врач', attribute='doctor', widget=ForeignKeyWidget(User, 'username'))

    class Meta:
        model = PatientAnswer


class PatientAnswerAdmin(ImportExportModelAdmin):
    resource_class = PatientAnswerResource

class OptionInline(admin.TabularInline):
    model = Option
    extra = 1

@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    inlines = [OptionInline]

class QuestionInline(admin.TabularInline):
    model = Question
    readonly_fields = ['question_text', 'proba']
    extra = 1

class NumericQuestionInline(admin.TabularInline):
    model = NumericQuestion
    extra = 1


class NumericOptionInline(admin.TabularInline):
    model = NumericOption
    extra = 1


@admin.register(NumericQuestion)
class NumericQuestionAdmin(admin.ModelAdmin):
    inlines = [NumericOptionInline]


admin.site.unregister(Question)
admin.site.register(Modality)
admin.site.unregister(NumericQuestion)
admin.site.register(Probs)
admin.site.register(Block)
admin.site.register(Question, QuestionAdmin)
admin.site.register(NumericQuestion, NumericQuestionAdmin)
admin.site.register(PatientNumericAnswer)
admin.site.register(PatientAnswer)
admin.site.register(ProbsImage)

