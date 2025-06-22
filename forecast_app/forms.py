from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField(
        label="Загрузите CSV-файл",
        help_text=".csv only",
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )

    def clean_file(self):
        f = self.cleaned_data['file']
        if not f.name.endswith('.csv'):
            raise forms.ValidationError("Неверный формат. Требуется .csv")
        return f

class ForecastParamForm(forms.Form):
    months_ahead = forms.IntegerField(label="Горизонт (лет)", min_value=1, max_value=5, initial=1)
    part_name = forms.ChoiceField(label="Запасная часть")
    method = forms.ChoiceField(
        label="Метод прогноза",
        choices=[
            ('ses', 'Экспоненциальное сглаживание'),
            ('ma', 'Скользящее среднее (MA)'),
            ('ar', 'Авторегрессия (AR)'),
            ('arima', 'ARIMA'),
            ('holt', 'Метод Хольта'),
            ('all', 'Все методы'),
        ],
        initial='ses'
    )

    def __init__(self, part_choices=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if part_choices:
            self.fields['part_name'].choices = [(p, p) for p in part_choices]