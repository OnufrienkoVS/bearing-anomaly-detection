import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from datetime import datetime
from pathlib import Path
from opcua import Client


from pipeline import SignalPreprocessingPipeline
from model import AnomalyDetector

st.set_page_config(
    page_title="Система обнаружения аномалий",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(":material/monitoring: Система обнаружения аномалий в вибрационном сигнале")
st.markdown("---")

if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'stream_processor' not in st.session_state:
    st.session_state.stream_processor = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

with st.sidebar:
    st.header(":material/settings: Настройки системы")
    
    st.subheader("Модель")
    model_path = st.text_input(
        "Путь к модели (.pth)",
        value="best_lstm-ae_ims.pth",
        key="model_path_input",
        help="Укажите путь к файлу с весами обученной модели"
    )
    
    threshold = st.number_input(
        "Порог обнаружения аномалий",
        value=1.478834e-08,
        format="%.2e",
        help="Чем ниже порог, тем чувствительнее система к аномалиям"
    )
    
    st.subheader("Обработка сигнала")
    window_size = st.number_input(
        "Размер окна (отсчетов)",
        value=1024,
        min_value=256,
        max_value=4096,
        step=256
    )
    
    window_overlap = st.slider(
        "Перекрытие окон",
        min_value=0.0,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Большее перекрытие даёт больше сегментов, но замедляет обработку"
    )
    
    seq_len = st.number_input(
        "Количество сегментов в одном пакете",
        value=10,
        min_value=2,
        max_value=32,
        step=5,
        help="Модель определяет аномалию не по одному сегменту сигнала, а по пакету сегментов, что увеличивает статистическую достоверность прогноза. Рекомендуется 5-10."
    )

    fs = st.number_input(
        "Частота дискретизации (Гц)",
        value=20000,
        min_value=1000,
        max_value=50000,
        step=1000,
        key="fs_main",
        help="Частота дискретизации входного сигнала"
    )
    
    st.markdown("---")
    if st.button(":material/build: Инициализировать систему", type="primary",  width='stretch'):
        try:
            st.session_state.pipeline = SignalPreprocessingPipeline(
                fs=fs,
                lowcut=10,
                highcut=5000,
                filter_order=4,
                window_size=window_size,
                window_overlap=window_overlap
            )
            
            st.session_state.detector = AnomalyDetector(
                model_path=model_path,
                threshold=threshold,
                input_dim=1,
                embedding_dim=32,
                seq_len=seq_len,
                window_size=window_size
            )
            
            st.success(":material/check: Система успешно инициализирована!")
            st.session_state.history = []
            
        except Exception as e:
            st.error(f":material/close: Ошибка инициализации: {str(e)}")
    
if st.session_state.detector is None:
    st.warning(":material/warning: Пожалуйста, настройте и инициализируйте систему в боковой панели")
    st.stop()

tab1, tab2, tab3= st.tabs([
    ":material/upload_file: Загрузка файла",
    ":material/api: OPC UA monitoring",
    ":material/history: История"
])

# Вкладка 1: Загрузка файла
with tab1:
    st.subheader("Загрузка файла с сигналом")
    
    # Две колонки только для загрузки и кнопки
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Выберите файл (ASCII, NPY)",
            type=['txt', 'npy'],
            help="Файл должен содержать временной ряд вибрационного сигнала"
        )
    
    errors, anomaly_flags = None, None

    with col2:
        if uploaded_file is not None:
            if st.button(":material/play_arrow: Запустить детекцию", type="primary", width='stretch'):
                with st.spinner("Обработка сигнала..."):
                    try:
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        
                        segments = st.session_state.pipeline.process_file(temp_path)
                        
                        if len(segments) >= st.session_state.detector.seq_len:
                            errors, anomaly_flags = st.session_state.detector.detect_batch(segments)
                            
                            # История
                            result = {
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'filename': uploaded_file.name,
                                'fs': fs,
                                'window_size': window_size,
                                'seq_len': seq_len,
                                'threshold': threshold,
                                'num_segments': len(segments),
                                'num_predictions': len(errors),
                                'mean_error': float(np.mean(errors)),
                                'max_error': float(np.max(errors)),
                                'anomaly_count': int(np.sum(anomaly_flags)),
                                'anomaly_rate': float(100 * np.sum(anomaly_flags) / len(errors)),
                                'errors': errors.tolist(),
                                'anomalies': anomaly_flags.tolist()
                            }
                            st.session_state.history.append(result)
                            
                            st.success(f":material/check: Обработка завершена!")
                            
                        else:
                            st.warning(f"Недостаточно данных: получено {len(segments)} сегментов, нужно {st.session_state.detector.seq_len}")
                        
                        os.remove(temp_path)
                        
                    except Exception as e:
                        st.error(f":material/close: Ошибка обработки: {str(e)}")

    if errors is not None and anomaly_flags is not None:
        # Метрики
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Всего пакетов", len(errors))
        with metric_col2:
            st.metric("Аномалий", np.sum(anomaly_flags))
        with metric_col3:
            st.metric("Доля аномалий", f"{100 * np.sum(anomaly_flags) / len(errors):.1f}%")
        with metric_col4:
            st.metric("Средняя ошибка", f"{np.mean(errors):.2e}")

        # График ошибок
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=errors,
            mode='lines',
            name='Ошибка реконструкции',
            line=dict(color='yellow', width=2)
        ))
                                
        fig.add_hline(
            y=threshold, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Порог: {threshold:.2e}"
        )
                                
        fig.update_layout(
            height=400,
            title="Ошибка реконструкции по пакетам",
            xaxis_title="Номер пакета",
            yaxis_title="MSE",
            yaxis_type="log"
        )
        st.plotly_chart(fig, width='stretch')

        # Таблица с аномалиями
        anomalies_df = pd.DataFrame({
            'Пакет': range(len(errors)),
            'Ошибка': errors,
            'Аномалия': anomaly_flags
        })
        anomalies_df = anomalies_df[anomalies_df['Аномалия'] == True]
                            
        if len(anomalies_df) > 0:
            st.subheader(f"Обнаруженные аномалии ({len(anomalies_df)})")
            st.dataframe(anomalies_df, width='stretch')

# Вкладка 2: OPC UA подключение
with tab2:
    st.subheader("OPC UA мониторинг")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        opc_url = st.text_input(
            "Endpoint URL",
            value="opc.tcp://localhost:4840",
            key="opc_client_url",
            help="Адрес OPC UA сервера"
        )
    
    with col2:
        if st.button("Подключиться", type="primary", width='stretch'):
            try:
                client = Client(opc_url)
                client.connect()
                
                objects = client.get_objects_node()
                vibration = objects.get_child(["2:Vibration"])
                
                st.session_state.opc_client = client
                st.session_state.opc_signal_node = vibration.get_child(["2:SignalArray"])
                st.session_state.opc_timestamp_node = vibration.get_child(["2:Timestamp"])
                st.session_state.opc_anomaly_node = vibration.get_child(["2:AnomalyFlag"])
                st.session_state.opc_error_node = vibration.get_child(["2:ReconstructionError"])
                
                st.success(":material/check: Подключено к OPC UA серверу")
                st.session_state.opc_connected = True
                st.session_state.opc_monitoring = False
                
            except Exception as e:
                st.error(f":material/close: Ошибка подключения: {str(e)}")
    
    with col3:
        if st.session_state.get('opc_connected', False):
            if st.button("Отключиться", width='stretch'):
                if st.session_state.get('opc_client'):
                    st.session_state.opc_client.disconnect()
                st.session_state.opc_connected = False
                st.session_state.opc_monitoring = False
                st.rerun()
    
    if st.session_state.get('opc_connected', False):
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            buffer_duration = st.number_input(
                "Время накопления (сек)",
                value=10,
                min_value=5,
                max_value=60,
                step=5,
                help="Время накопления данных перед детекцией"
            )
        
        with col2:
            if not st.session_state.get('opc_monitoring', False):
                if st.button(":material/play_arrow: Начать мониторинг", type="primary", width='stretch'):
                    st.session_state.opc_monitoring = True
                    st.session_state.opc_data_buffer = []
                    st.session_state.opc_last_timestamp = None
                    st.session_state.opc_cycle_results = []
                    st.rerun()
            else:
                if st.button(":material/stop: Остановить", type="secondary", width='stretch'):
                    st.session_state.opc_monitoring = False
                    st.rerun()
        
        if st.session_state.get('opc_monitoring', False):
            try:
                current_timestamp = st.session_state.opc_timestamp_node.get_value()
                signal_array = st.session_state.opc_signal_node.get_value()
                
                if signal_array and len(signal_array) > 0:
                    if st.session_state.opc_last_timestamp != current_timestamp:
                        st.session_state.opc_last_timestamp = current_timestamp
                        current_signal = np.array(signal_array)
                        st.session_state.opc_data_buffer.append({
                            'timestamp': current_timestamp,
                            'data': current_signal
                        })
                        buffer_size = len(st.session_state.opc_data_buffer)
                        target_size = buffer_duration
                        progress = min(1.0, buffer_size / target_size)
                        st.progress(progress)
                        st.caption(f":material/package_2: Накоплено: {buffer_size} / {target_size} секунд ({progress*100:.1f}%)")
                    
                        if buffer_size >= target_size:
                            full_signal = []
                            for item in st.session_state.opc_data_buffer:
                                full_signal.extend(item['data'].tolist())
                            signal = np.array(full_signal)
                            segments = st.session_state.pipeline.process(signal)
                            
                            errors, anomaly_flags = st.session_state.detector.detect_batch(segments)
                        
                            if len(errors) > 0:
                                # Результаты цикла
                                cycle_result = {
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'data_start': st.session_state.opc_data_buffer[0]['timestamp'],
                                    'data_end': st.session_state.opc_data_buffer[-1]['timestamp'],
                                    'duration': buffer_duration,
                                    'num_predictions': len(errors),
                                    'mean_error': float(np.mean(errors)),
                                    'anomaly_count': int(np.sum(anomaly_flags)),
                                    'anomaly_rate': float(100 * np.sum(anomaly_flags) / len(errors))
                                }
                                st.session_state.opc_cycle_results.append(cycle_result)

                                # История
                                history_result = {
                                    'timestamp': cycle_result['timestamp'],
                                    'filename': f"OPC UA ({cycle_result['data_start']})",
                                    'num_predictions': cycle_result['num_predictions'],
                                    'anomaly_count': cycle_result['anomaly_count'],
                                    'anomaly_rate': cycle_result['anomaly_rate'],
                                    'mean_error': cycle_result['mean_error']
                                }
                                st.session_state.history.append(history_result)

                                # Отправка в OPC UA
                                try:
                                    st.session_state.opc_anomaly_node.set_value(bool(any(anomaly_flags)))
                                    st.session_state.opc_error_node.set_value(float(np.mean(errors)))
                                except:
                                    pass
                            
                            
                                # Метрики
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Предсказаний", cycle_result['num_predictions'])
                                with col2:
                                    st.metric("Аномалий", cycle_result['anomaly_count'])
                                with col3:
                                    st.metric("Доля аномалий", f"{cycle_result['anomaly_rate']:.1f}%")
                                with col4:
                                    st.metric("Средняя ошибка", f"{cycle_result['mean_error']:.2e}")
                            
                                # График ошибок за последний цикл
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    y=errors,
                                    mode='lines',
                                    name='Ошибка реконструкции',
                                    line=dict(color='yellow', width=2)
                                ))

                                fig.add_hline(
                                    y=threshold, 
                                    line_dash="dash", 
                                    line_color="red",
                                    annotation_text=f"Порог: {threshold:.2e}"
                                )

                                fig.update_layout(
                                    height=300,
                                    title=f"Ошибка реконструкции",
                                    xaxis_title="Номер предсказания",
                                    yaxis_title="MSE",
                                    yaxis_type="log"
                                )
                                st.plotly_chart(fig, width='stretch')
                            
                            st.session_state.opc_data_buffer = []
                        
                            time.sleep(1)
                            st.rerun()
                    
            except Exception as e:
                st.error(f":material/close: Ошибка: {str(e)}")
                time.sleep(1)
            
            time.sleep(0.5)
            st.rerun()
        
        if len(st.session_state.get('opc_cycle_results', [])) > 0:
            st.markdown("---")
            st.subheader("Сводка по мониторингу")
            
            summary_df = pd.DataFrame([
                {
                    'Время': r['timestamp'],
                    'Предсказаний': r['num_predictions'],
                    'Аномалий': r['anomaly_count'],
                    'Доля': f"{r['anomaly_rate']:.1f}%",
                    'Ср. ошибка': f"{r['mean_error']:.2e}"
                }
                for r in st.session_state.opc_cycle_results[-10:]
            ])
            st.dataframe(summary_df, width='stretch')
            
            if st.button(":material/delete: Очистить историю мониторинга", width='stretch'):
                st.session_state.opc_cycle_results = []
                st.session_state.opc_data_buffer = []
                st.session_state.opc_last_timestamp = None
                st.rerun()

# Вкладка 3: История
with tab3:
    st.subheader("История обработки")
    
    if len(st.session_state.history) == 0:
        st.info("Нет данных в истории. Загрузите файл или запустите OPC UA мониторинг.")
    else:
        history_df = pd.DataFrame([
            {
                'Время': h['timestamp'],
                'Источник': h['filename'],
                'Предсказаний': h['num_predictions'],
                'Аномалий': h['anomaly_count'],
                'Доля аномалий': f"{h['anomaly_rate']:.1f}%",
                'Ср. ошибка': f"{h['mean_error']:.2e}"
            }
            for h in st.session_state.history
        ])
        
        st.dataframe(history_df, width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(":material/delete: Очистить историю", width='stretch'):
                st.session_state.history = []
                st.rerun()
        
        with col2:
            if len(st.session_state.history) > 0:
                export_df = pd.DataFrame([
                    {
                        'Время': h['timestamp'],
                        'Источник': h['filename'],
                        'Предсказаний': h['num_predictions'],
                        'Аномалий': h['anomaly_count'],
                        'Доля_аномалий': h['anomaly_rate'],
                        'Средняя_ошибка': h['mean_error']
                    }
                    for h in st.session_state.history
                ])
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label=":material/download: Экспортировать CSV",
                    data=csv,
                    file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    width='stretch'
                )