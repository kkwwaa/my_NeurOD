from amscsait import settings
from . import views
from django.urls import path, include

urlpatterns = [
    path("", views.index, name="index"),
    path("admin/", views.admin, name="admin"),
    path("create_proba", views.create_proba, name="create_proba"),
    path("create_patient", views.create_patient, name="create_patient"),
    path("patient/<int:pk>", views.view_patient, name="view_patient"),
    path("patient/<int:pk>/edit", views.edit_patient, name="edit_patient"),
    path("patient/<int:pk>/probs_list", views.probs_list, name="probs_list"),
    path("patient/<int:pk>/probs_list/<int:proba_pk>", views.proba, name="proba"),
    path("patient/<int:pk>/probs_results/<int:modal>", views.probs_results, name="probs_results"),
    path("register", views.register, name="register"),
    path("login", views.user_login, name="login"),
    path("exit", views.exit, name="exit"),
]

if settings.DEBUG:
    import debug_toolbar

    urlpatterns = [
                      path('__debug__/', include(debug_toolbar.urls)),
                  ] + urlpatterns
