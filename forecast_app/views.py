import base64
import csv
import datetime
import io

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.template.loader import render_to_string
from plotly.offline import plot
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from xhtml2pdf import pisa
from xhtml2pdf.default import DEFAULT_FONT

from .forms import UploadFileForm, ForecastParamForm


def upload_file(request):
    context = {}
    preview_df = None

    # Очистка сессии
    if request.GET.get("clear") == "1":
        keys_to_clear = [
            "data", "preview_table",
            "forecast_table", "forecast_part",
            "forecast_plot", "forecast_method",
            "forecast_chart_base64", "forecast_errors"
        ]
        for key in keys_to_clear:
            request.session.pop(key, None)
        context["success"] = "Файл удалён."
        context["form"] = UploadFileForm()
        return render(request, "upload.html", context)

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        errors = []

        if form.is_valid():
            file = form.cleaned_data['file']

            # Проверка расширения файла
            if not file.name.lower().endswith('.csv'):
                errors.append("Файл должен быть в формате CSV.")
            else:
                # Попытка чтения CSV
                try:
                    df = pd.read_csv(file)
                except Exception as e:
                    errors.append(f"Ошибка чтения CSV: {str(e)}")
                else:
                    # Проверка наличия столбцов с годами
                    year_cols = [col for col in df.columns if col.startswith("y")]
                    if not year_cols:
                        errors.append("Отсутствуют столбцы с данными по годам (начинаются с 'y').")
                    # Проверка на пропущенные значения
                    if df.isnull().values.any():
                        errors.append("В файле есть пропущенные значения.")
                    # Проверка на дубликаты
                    if df.duplicated().any():
                        errors.append("В файле есть дубликаты строк.")

            if errors:
                context['errors'] = errors
            else:
                # Сохранение данных и формирование предпросмотра
                request.session['data'] = df.to_json()
                context['success'] = "Файл успешно загружен!"
                years = [col[1:] for col in year_cols]
                preview_df = df[['name'] + year_cols].copy()
                preview_df.columns = ['Запасная часть'] + years
                context['preview_table'] = preview_df.to_html(index=False, classes="table table-bordered")
                request.session['preview_table'] = context['preview_table']
        else:
            # Сбор ошибок валидации формы
            for field, field_errors in form.errors.items():
                for error in field_errors:
                    errors.append(f"{error}")
            context['errors'] = errors

    else:
        form = UploadFileForm()

    context['form'] = form
    context['preview_table'] = request.session.get('preview_table')
    return render(request, 'upload.html', context)


def forecast_view(request):
    only_history = request.POST.get('only_history') == 'on'
    if 'data' not in request.session:
        return render(request, 'forecast.html', {'error': 'Сначала загрузите CSV файл.'})

    df = pd.read_json(io.StringIO(request.session['data']))

    year_columns = [col for col in df.columns if col.startswith('y') and col[1:].isdigit()]
    if not year_columns:
        return render(request, 'forecast.html', {'error': 'Нет годовых столбцов вида y2016, y2017 и т.д.'})

    part_choices = df['name'].tolist()
    forecast_result = None
    plot_div = None

    if request.method == 'POST':
        form = ForecastParamForm(data=request.POST, part_choices=part_choices)
        if form.is_valid():
            part = form.cleaned_data['part_name']
            years_forward = form.cleaned_data['months_ahead']
            method = form.cleaned_data['method']
            alpha = request.session.get('alpha', 0.5)
            error_years = request.session.get("error_years", 1)

            try:
                row = df[df['name'] == part].iloc[0]
                consumption = row[year_columns]

                years = [int(col[1:]) for col in year_columns]
                date_index = pd.to_datetime([f"{y}-01-01" for y in years])
                series = pd.Series(consumption.values, index=date_index).astype(float).dropna().asfreq('YS')

                if only_history:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=series.index.year,
                        y=series.values,
                        mode='lines+markers',
                        name='Исторические данные',
                        line=dict(color='black'),
                        marker=dict(symbol='circle'),
                        hovertemplate='%{y:.0f} шт<br>Год: %{x}<extra></extra>'
                    ))
                    fig.update_layout(
                        title=f'История потребления: {part}',
                        xaxis_title='Год',
                        yaxis_title='Расход, шт.',
                        template='plotly_white',
                        hovermode='x unified',
                        xaxis=dict(
                            tickmode='array',
                            tickvals=series.index.year,
                            ticktext=[str(y) for y in series.index.year],
                            tickangle=+45,
                            automargin=True,
                        )
                    )
                    plot_div = plot(fig, output_type='div')
                    return render(request, 'forecast.html', {
                        'form': form,
                        'plot_div': plot_div,
                        'table_html': None,
                    })

                if len(series) < 3:
                    return render(request, 'forecast.html', {
                        'form': form,
                        'error': f"Недостаточно данных для прогноза (нужно ≥ 3, сейчас {len(series)})."
                    })

                forecasts = {}
                forecast_horizon_index = pd.date_range(
                    start=series.index[-1] + pd.DateOffset(years=1),
                    periods=years_forward,
                    freq='YS'
                ).to_period('Y').to_timestamp()

                if method in ('ses', 'all'):
                    fit = SimpleExpSmoothing(series).fit(smoothing_level=alpha, optimized=False)
                    forecasts['SES'] = fit.forecast(years_forward)

                if method in ('ma', 'all'):
                    ma_value = series.rolling(window=3).mean().iloc[-1]
                    forecasts['MA'] = pd.Series([ma_value] * years_forward, index=forecast_horizon_index)

                if method in ('ar', 'all'):
                    fit = AutoReg(series, lags=2).fit()
                    pred = fit.predict(start=len(series), end=len(series) + years_forward - 1)
                    pred.index = forecast_horizon_index
                    forecasts['AR'] = pred

                if method in ('arima', 'all'):
                    fit = ARIMA(series, order=(2, 0, 0)).fit()
                    pred = fit.forecast(steps=years_forward)
                    pred.index = forecast_horizon_index
                    forecasts['ARIMA'] = pred

                if method in ('holt', 'all'):
                    fit = Holt(series, initialization_method="estimated").fit(
                        smoothing_level=alpha,
                        smoothing_trend=0.3,
                        optimized=False)
                    forecasts['Holt'] = fit.forecast(years_forward)
                    forecasts['Holt'].index = forecast_horizon_index

                fig = go.Figure()
                # График: Plotly
                actual_trace = go.Scatter(
                    x=series.index.year,
                    y=series.values,
                    mode='lines+markers',
                    name='Исторические данные',
                    line=dict(color='black'),
                    marker=dict(symbol='circle'),
                    hovertemplate='%{y:.0f} шт<br>Год: %{x}<extra></extra>'
                )

                fig.add_trace(actual_trace)

                colors = ['blue', 'orange', 'green', 'red', 'purple']
                if not only_history and forecasts:
                    for idx, (name, forecast) in enumerate(forecasts.items()):
                        color = colors[idx % len(colors)]
                        # Соединительная линия
                        fig.add_trace(go.Scatter(
                            x=[series.index[-1].year, forecast.index[0].year],
                            y=[series.values[-1], forecast.iloc[0]],
                            mode='lines',
                            line=dict(color=color, dash='dash'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        # Линия прогноза
                        fig.add_trace(go.Scatter(
                            x=forecast.index.year,
                            y=forecast.round(0),
                            mode='lines+markers',
                            name=name,
                            line=dict(dash='dash', color=color),
                            marker=dict(symbol='circle', color=color),
                            hovertemplate='%{y:.0f} шт<br>Год: %{x}<extra></extra>'
                        ))

                # Получаем все годы из исторических + прогнозных данных
                all_years = list(series.index.year)  # история
                for fc in forecasts.values():
                    all_years.extend(fc.index.year)
                all_years = sorted(set(all_years))  # уникальные и по порядку

                fig.update_layout(
                    title=f'Прогноз потребления: {part}',
                    xaxis_title='Год',
                    yaxis_title='Расход, шт.',
                    template='plotly_white',
                    hovermode='x unified',

                    xaxis = dict(
                        tickmode='array',
                        tickvals=all_years,
                        ticktext=[str(y) for y in all_years],
                        tickangle=+45,
                        automargin=True,
                    )
                )

                plot_div = plot(fig, output_type='div')

                # Сохранение графика для PDF
                png_bytes = pio.to_image(fig, format='png')
                request.session['forecast_chart_base64'] = base64.b64encode(png_bytes).decode()

                # Формирование таблицы
                combined_df = pd.DataFrame(index=forecast_horizon_index)
                combined_df['Год'] = combined_df.index.year
                names = []
                for name, fcst in forecasts.items():
                    combined_df[name] = fcst.round(0).astype(int)
                    names.append(name)
                print(names)
                forecast_result = combined_df
                print(forecast_result)

                used_methods = list(forecasts.keys())
                method_mapping = {
                    'SES': 'Экспоненциальное сглаживание',
                    'MA': 'Скользящее среднее (MA)',
                    'AR': 'Авторегрессия (AR)',
                    'ARIMA': 'ARIMA',
                    'Holt': 'Метод Хольта'
                }
                request.session['forecast_methods'] = [method_mapping.get(m, m) for m in used_methods]

                # Сохранение данных в сессию
                request.session['forecast_table'] = combined_df.to_dict('records')
                request.session['forecast_part'] = part
                request.session['forecast_method'] = method
                request.session['forecast_plot'] = plot_div

                # Кол-во лет для теста
                error_periods = min(error_years, years_forward, len(series))
                if error_periods == 0:
                    return render(request, 'forecast.html', {
                        'form': form,
                        'error': "Недостаточно данных для сравнения. Уменьшите глубину анализа ошибок или увеличьте горизонт прогноза."
                    })

                # Синхронизация
                y_true = series[-error_periods:]
                y_pred = forecast[:error_periods]
                y_pred.index = y_true.index

                # Метрики
                def rmse(y_true, y_pred):
                    return np.sqrt(np.mean((y_true - y_pred) ** 2))

                def mape(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

                def smape(y_true, y_pred):
                    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

                def wape(y_true, y_pred):
                    return 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

                errors = {}

                for method_name, forecast_series in forecasts.items():
                    y_pred = forecast_series[:error_periods].astype(float)
                    y_pred.index = y_true.index  # синхронизация индексов

                    errors[method_name] = {
                        'RMSE': round(rmse(y_true, y_pred), 2),
                        'MAPE': round(mape(y_true, y_pred), 2),
                        'SMAPE': round(smape(y_true, y_pred), 2),
                        'WAPE': round(wape(y_true, y_pred), 2),
                    }

                # Сохраняем в сессию
                request.session['forecast_errors'] = errors

            except Exception as e:
                return render(request, 'forecast.html', {
                    'form': form,
                    'error': f"Ошибка построения прогноза: {str(e)}"
                })

    else:
        form = ForecastParamForm(part_choices=part_choices)

    return render(request, 'forecast.html', {
        'form': form,
        'plot_div': plot_div,
        'table_html': forecast_result.to_html(index=False) if forecast_result is not None else None,
    })

def reports_view(request):
    forecast_data = request.session.get('forecast_table')
    part = request.session.get('forecast_part')
    method_names = request.session.get('forecast_methods', ['N/A'])
    plot_html = request.session.get('forecast_plot')
    errors = request.session.get('forecast_errors', {})

    if not forecast_data or not part:
        return render(request, 'reports.html', {'error': 'Нет данных для отчета. Сначала постройте прогноз.'})

    forecast_df = pd.DataFrame(forecast_data)

    if request.GET.get('download') == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="forecast_{part}.csv"'
        response.write('\ufeff')

        writer = csv.writer(response)
        writer.writerow(['Год', 'Прогноз'])
        for row in forecast_df.itertuples(index=False):
            writer.writerow(row)

        return response

    if request.GET.get('download') == 'pdf':
        pdfmetrics.registerFont(TTFont("DejaVuSans", "forecast_app/static/fonts/DejaVuSans.ttf"))
        DEFAULT_FONT["helvetica"] = "DejaVuSans"
        include_chart = request.GET.get('chart') == '1'
        chart_base64 = request.session.get('forecast_chart_base64') if include_chart else None

        html = render_to_string('report_pdf.html', {
            'table': forecast_df.to_html(index=False, classes="table table-bordered"),
            'part': part,
            'method': ', '.join(method_names),
            'date': datetime.datetime.now(),
            'errors': errors,
            'chart_base64': chart_base64,
        })

        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="forecast_{part}.pdf"'
        pisa_status = pisa.CreatePDF(html, dest=response, encoding='utf-8')
        if pisa_status.err:
            return HttpResponse('Ошибка генерации PDF')
        return response

    return render(request, 'reports.html', {
        'table': forecast_df.to_html(index=False, classes="table table-bordered"),
        'part': part,
        'method': method_names,
        'errors': errors,
        'plot_div': plot_html
    })


def settings_view(request):
    if request.method == 'POST':
        request.session['alpha'] = float(request.POST.get('alpha', 0.5))
        request.session['error_years'] = int(request.POST.get('error_years', 3))
        return redirect('settings')

    context = {
        'alpha': request.session.get('alpha', 0.5),
        'error_years': request.session.get('error_years', 3),
    }
    return render(request, 'settings.html', context)
