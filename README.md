# Обнаружение аномалий в вибрационных сигналах подшипников качения

## Описание

Данный репозиторий содержит исходный код выпускной квалификационной работы (бакалавра) на тему: **"Разработка ML-модели машинного обучения для обнаружения аномалий в работе промышленного оборудования"**

## Структура репозитория
```
bearing-anomaly-detection/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── notebooks/
│ ├── 1_preprocessing.ipynb     # Загрузка данных, фильтрация, сегментация, извлечение признаков
│ └── 2_train&eval.ipynb        # Обучение моделей и оценка
│
├── prototype/
│ ├── app.py                    # Streamlit веб-интерфейс
│ ├── model.py                  # LSTM-AE
│ ├── pipeline.py               # Предобработка сигнала
│ ├── opc_server.py             # OPC server + симулятор датчика вибрации
│ └── test_data/                # Примеры сигналов для тестирования
│   ├── normal.txt
│   └── defect.txt
│
└── saved_models/
  ├── best_ae_cwru.pth          # Полносвязный AE (обучен на CWRU)
  ├── best_ae_ims.pth           # Полносвязный AE (обучен на IMS)
  ├── best_lstm-ae_cwru.pth     # LSTM-AE (обучен на CWRU)
  └── best_lstm-ae_ims.pth      # LSTM-AE (обучен на IMS)
```

## Датасеты

В работе использованы два публичных набора данных:

| Датасет | Описание | Источник |
|---------|----------|----------|
| **CWRU** | Вибрационные сигналы подшипников с искусственными дефектами | [Case Western Reserve University](https://engineering.case.edu/bearingdatacenter/welcome) |
| **IMS** | Процесс деградации подшипника до отказа | [NASA Ames Research Center](https://data.nasa.gov/dataset/ims-bearings) |

> **Примечание:** Исходные файлы датасетов не включены в репозиторий из-за большого объёма. Перед запуском ноутбуков необходимо скачать датасеты и поместить их в соответствующие директории согласно инструкциям в `1_Preprocessing.ipynb`.

## Модели

В рамках работы реализованы и сравнены следующие модели: Isolation Forest, One-Class SVM, Autoencoder, LSTM-autoencoder

## Результаты (указать результаты)

### Набор данных CWRU

| Модель | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| Isolation Forest | 0.xx | 0.xx | 0.xx |
| One-Class SVM | 0.xx | 0.xx | 0.xx |
| Autoencoder | 0.xx | 0.xx | 0.xx |
| LSTM-Autoencoder | 0.xx | 0.xx | 0.xx |

### Набор данных IMS

| Модель | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| Isolation Forest | 0.xx | 0.xx | 0.xx |
| One-Class SVM | 0.xx | 0.xx | 0.xx |
| Autoencoder | 0.xx | 0.xx | 0.xx |
| LSTM-Autoencoder | 0.xx | 0.xx | 0.xx |