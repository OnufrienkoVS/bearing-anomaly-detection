import numpy as np
from scipy.signal import butter, filtfilt
from typing import List

class SignalPreprocessingPipeline:
    """
    Пайплайн предобработки вибрационного сигнала:
    1. Фильтрация
    2. Сегментация
    """
    
    def __init__(self, 
                 fs: int = 20000,
                 lowcut: float = 10,
                 highcut: float = 5000,
                 filter_order: int = 4,
                 window_size: int = 1024,
                 window_overlap: float = 0.5):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.window_size = window_size
        self.window_overlap = window_overlap
        
        self._init_filter()
    
    def _init_filter(self):
        """Инициализация полосового фильтра Баттерворта"""
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        self.b, self.a = butter(self.filter_order, [low, high], btype='band')
    
    def apply_filter(self, signal: np.ndarray) -> np.ndarray:
        """Применение полосового фильтра к сигналу"""
        return filtfilt(self.b, self.a, signal)
    
    def segment_signal(self, signal: np.ndarray) -> np.ndarray:
        """Сегментация сигнала с перекрытием"""
        step = int(self.window_size * (1 - self.window_overlap))
        
        segments = []
        for start in range(0, len(signal) - self.window_size + 1, step):
            segment = signal[start:start + self.window_size]
            segments.append(segment)
        
        return np.array(segments)
    
    def process(self, signal: np.ndarray) -> np.ndarray:
        """Основной метод обработки сигнала"""
        if isinstance(signal, list):
            signal = np.array(signal)
        
        if len(signal.shape) != 1:
            raise ValueError(f"Ожидается 1D сигнал, получена форма {signal.shape}")
        
        # Фильтрация
        filtered_signal = self.apply_filter(signal)
        
        # Сегментация
        return self.segment_signal(filtered_signal)
    
    def process_file(self, 
                     file_path: str, 
                     column: int = 0,
                     delimiter: str = '\t') -> np.ndarray:
        """Чтение и обработка сигнала из файла"""
        # Определение формата файла
        if file_path.endswith('.npy'):
            signal = np.load(file_path)
            if len(signal.shape) > 1:
                signal = signal[:, column] if column < signal.shape[1] else signal.flatten()
        else:
            # Чтение ASCII
            data = []
            with open(file_path, 'r', encoding='ascii') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(delimiter)
                        if column < len(parts):
                            data.append(float(parts[column]))
            signal = np.array(data)
        
        return self.process(signal)

class StreamingProcessor:
    """Класс для потоковой обработки сигнала в реальном времени"""
    
    def __init__(self, pipeline: SignalPreprocessingPipeline):
        self.pipeline = pipeline
        self.buffer = np.array([])
        
    def process_chunk(self, chunk: np.ndarray) -> List[np.ndarray]:
        """
        Обработка очередного чанка данных.
        Возвращает список новых сегментов, полученных из этого чанка.
        """
        # Добавляем новые данные в буфер
        self.buffer = np.concatenate([self.buffer, chunk]) if len(self.buffer) > 0 else chunk
        new_segments = []
        step = int(self.pipeline.window_size * (1 - self.pipeline.window_overlap))
        
        # Пока в буфере есть хотя бы одно полное окно, выделяем его и предобрабатываем
        while len(self.buffer) >= self.pipeline.window_size:
            window = self.buffer[:self.pipeline.window_size]
            filtered = self.pipeline.apply_filter(window)
            new_segments.append(filtered)
            self.buffer = self.buffer[step:]

        return np.array(new_segments) if new_segments else np.array([])
    
    def reset(self):
        """Сброс буфера"""
        self.buffer = np.array([])


# Пример использования для тестирования файла
if __name__ == "__main__":
    import os
    
    # Создание пайплайна
    pipeline = SignalPreprocessingPipeline(
        fs=20000,
        lowcut=10,
        highcut=5000,
        window_size=1024,
        window_overlap=0.5
    )
    
    # Обработка реального файла IMS
    print("\n" + "=" * 60)
    print("ТЕСТ 1: Обработка реального файла IMS")
    print("=" * 60)
    
    # Путь к файлу
    file_path = "../IMS/raw_ASCII/2004.02.12.10.32.39"
    
    # Проверка существования файла
    if os.path.exists(file_path):
        print(f"Файл найден: {file_path}")
        
        # Обработка файла
        segments = pipeline.process_file(file_path, column=0, delimiter='\t')
        
        print(f"Сегментов получено: {len(segments)}")
        print(f"Форма сегментов: {segments.shape}")
        
        # Тест потоковой обработки
        print("\n" + "=" * 60)
        print("ТЕСТ 2: Потоковая обработка (имитация реального времени)")
        print("=" * 60)
        
        # Читаем файл заново для потокового теста
        data = []
        with open(file_path, 'r', encoding='ascii') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) > 0:
                        data.append(float(parts[0]))
        
        full_signal = np.array(data)
        print(f"Полный сигнал: {len(full_signal)} отсчетов")
        
        # Создаём потоковый процессор
        stream_processor = StreamingProcessor(pipeline)
        
        # Имитируем получение данных чанками по 5000 отсчетов
        chunk_size = 5000
        all_stream_segments = []
        
        for i in range(0, len(full_signal), chunk_size):
            chunk = full_signal[i:i+chunk_size]
            new_segments = stream_processor.process_chunk(chunk)
            if len(new_segments) > 0:
                all_stream_segments.extend(new_segments)
                print(f"  Чанк {i//chunk_size + 1}: +{len(new_segments)} сегментов, всего: {len(all_stream_segments)}")
        
        print(f"\nИтого сегментов (потоковый режим): {len(all_stream_segments)}")
        
        # Сравнение с пакетной обработкой
        print(f"\nПакетная обработка дала: {len(segments)} сегментов")
        print(f"Потоковая обработка дала: {len(all_stream_segments)} сегментов")
        
        if len(segments) == len(all_stream_segments):
            print("Результаты совпадают!")
        else:
            print(f"Расхождение: {abs(len(segments) - len(all_stream_segments))} сегментов")
            
    else:
        print(f"Файл не найден: {file_path}")
        print("Проверьте путь к файлу. Текущая директория:", os.getcwd())