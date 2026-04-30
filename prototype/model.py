import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from torch.utils.data import DataLoader, TensorDataset

class LSTM_Autoencoder(nn.Module):
    def __init__(self, input_size=1, embedding_dim=32):
        super(LSTM_Autoencoder, self).__init__()
        
        # Энкодер
        self.encoder_lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=embedding_dim*4,
            batch_first=True
        )
        self.encoder_lstm2 = nn.LSTM(
            input_size=embedding_dim*4,
            hidden_size=embedding_dim*2,
            batch_first=True
        )
        self.encoder_lstm3 = nn.LSTM(
            input_size=embedding_dim*2,
            hidden_size=embedding_dim,
            batch_first=True
        )
        
        # Декодер
        self.decoder_lstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True
        )
        self.decoder_lstm2 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim*2,
            batch_first=True
        )
        self.decoder_lstm3 = nn.LSTM(
            input_size=embedding_dim*2,
            hidden_size=embedding_dim*4,
            batch_first=True
        )
        
        # Выходной слой
        self.output_layer = nn.Linear(embedding_dim*4, input_size)
        self.activation = nn.ELU()
        
    def forward(self, x):
        # Энкодинг
        out, _ = self.encoder_lstm1(x)
        out = self.activation(out)
        
        out, _ = self.encoder_lstm2(out)
        out = self.activation(out)

        out, (h_n, _) = self.encoder_lstm3(out)
        
        # Декодинг
        out, _ = self.decoder_lstm1(out)
        out = self.activation(out)
        
        out, _ = self.decoder_lstm2(out)
        out = self.activation(out)

        out, _ = self.decoder_lstm3(out)
        out = self.activation(out)
        
        # Выходной слой
        output = self.output_layer(out)
        
        return output
    
    def compute_reconstruction_error(self, x: torch.Tensor) -> np.ndarray:
        """Вычисление ошибки реконструкции для батча последовательностей"""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = nn.MSELoss(reduction='none')(reconstructed, x)
            error = mse.mean(dim=(1, 2)).cpu().numpy()
        return error

class DataFormatter:
    """Функции для форматирования данных для LSTM-AE"""
    @staticmethod
    def create_dataloader(segments: np.ndarray,
                          window_size=1024,
                          batch_size: int = 32,
                          shuffle: bool = False) -> DataLoader:
        """Создание DataLoader из сегментов"""
        X = segments.reshape(-1, window_size, 1)
        dataset = TensorDataset(torch.FloatTensor(X))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    @staticmethod
    def create_sequence_from_segments(segments: List[np.ndarray]) -> torch.Tensor:
        """Создание тензора последовательности из списка сегментов"""
        window_size = len(segments[0])
        segments = np.array(segments)
        sequence = segments.reshape(-1, window_size, 1)
        sequence = torch.FloatTensor(sequence)
        return sequence

    
class AnomalyDetector:
    """Детектор аномалий на основе LSTM-AE."""
    
    def __init__(self, 
                 model_path: str, 
                 threshold: float, 
                 input_dim: int = 1, 
                 embedding_dim: int = 32, 
                 seq_len: int = 10,
                 window_size: int = 10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.threshold = threshold
        self.window_size = window_size
        
        # Загрузка модели
        self.model = LSTM_Autoencoder(input_size=input_dim, embedding_dim=embedding_dim)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Буфер для накопления сегментов
        self.buffer: List[np.ndarray] = []
        
        # Форматтер данных
        self.formatter = DataFormatter()
    
    def add_segment(self, segment: np.ndarray) -> Optional[Tuple[bool, float]]:
        """Добавление нового сегмента в буфер и детекция аномалии."""
        # Проверка формы
        if len(segment.shape) == 1:
            if segment.shape[0] != self.window_size:
                print(f"Warning: segment shape {segment.shape} != expected {self.window_size}")
        elif len(segment.shape) == 2:
            segment = segment.flatten()
        
        # Добавляем в буфер
        self.buffer.append(segment)
        
        # Оставляем только последние seq_len сегментов
        if len(self.buffer) > self.seq_len:
            self.buffer = self.buffer[-self.seq_len:]
        
        # Если накоплено достаточно данных
        if len(self.buffer) == self.seq_len:
            sequence = self.formatter.create_sequence_from_segments(self.buffer)
            sequence = sequence.to(self.device)
            errors = self.model.compute_reconstruction_error(sequence)
            
            return np.mean(errors) > self.threshold, errors
        
        return None
    
    def detect_batch(self, segments: np.ndarray) -> np.ndarray:
        """Пакетная детекция аномалий"""
        dataloader = self.formatter.create_dataloader(segments, window_size=self.window_size, batch_size=self.seq_len)
        
        errors = []
        for batch in dataloader:
            # Берём пакет сегментов
            X = batch[0].to(self.device)
            error = self.model.compute_reconstruction_error(X)
            errors.append(np.mean(error))   # Среднии ошибки по батчу
        errors = np.array(errors)
        return errors, errors > self.threshold
    
    def predict_from_signal(self, signal: np.ndarray, pipeline) -> np.ndarray:
        """Полный пайплайн: сигнал -> обработанный сигнал -> детекция"""
        # Обработка сигнала
        segments = pipeline.process(signal)
        return self.detect_batch(segments)
    
    def reset(self):
        """Сброс буфера"""
        self.buffer = []
    
    def set_threshold(self, threshold: float):
        self.threshold = threshold
        print(f"Порог обновлён: {threshold:.2e}")
    
    def get_buffer_status(self) -> dict:
        return {
            "current_buffer_size": len(self.buffer),
            "required_size": self.seq_len,
            "ready": len(self.buffer) == self.seq_len
        }
    
# Пример использования
if __name__ == "__main__":
    from pipeline import SignalPreprocessingPipeline
    
    # Параметры
    MODEL_PATH = "best_lstm-ae_ims.pth"
    THRESHOLD = 1.478834e-08
    WINDOW_SIZE = 1024
    SEQ_LEN = 10

    # Файлы для теста 3
    NORMAL_FILE_PATH = "../IMS/raw_ASCII/2004.02.12.12.02.39"  # Путь к нормальному файлу
    FAULT_FILE_PATH = "../IMS/raw_ASCII/2004.02.18.14.52.39"   # Путь к файлу с дефектом
    
    # Создаём пайплайн
    pipeline = SignalPreprocessingPipeline(
        fs=20000,
        lowcut=10,
        highcut=5000,
        window_size=WINDOW_SIZE,
        window_overlap=0.5
    )
    
    # Создаём детектор
    detector = AnomalyDetector(
        model_path=MODEL_PATH,
        threshold=THRESHOLD,
        seq_len=SEQ_LEN,
        window_size=WINDOW_SIZE
    )
    
    # Генерация тестового сигнала
    t = np.linspace(0, 1, WINDOW_SIZE)
    signal1 = np.sin(2 * np.pi * 100 * t)
    signal2 = signal1.copy()
    signal2[500:550] += 5.0
    
    # Тест потокового режима
    print("Тест 1 (потокового режима):")
    detector.reset()
    
    # Заполняем буфер
    for i in range(SEQ_LEN - 1):
        detector.add_segment(signal1)

    # Сигнал 1
    result = detector.add_segment(signal2)
    if result:
        is_anomaly, error = result
        print(f"Сигнал 1: error={np.mean(error):.2e}, anomaly={is_anomaly}")
    
    # Сигнал 2
    result = detector.add_segment(signal2)
    if result:
        is_anomaly, error = result
        print(f"Сигнал 2: error={np.mean(error):.2e}, anomaly={is_anomaly}")
    
    # Тест пакетного режима
    print("\nТест 2 (пакетного режима):")
    segments = np.array([signal1] * 50)
    errors, res = detector.detect_batch(segments)
    print(f"  Обработано {len(errors)} окон, ошибки: mean={errors.mean():.2e}, аномалии: {res}")

    # Тест реальных файлов набора IMS
    print("\nТест 3 (IMS файлов):")
    print('Нормальный файл:')
    segments = pipeline.process_file(NORMAL_FILE_PATH)
    errors, res = detector.detect_batch(segments)
    print(f"Обработано {len(errors)} окон (батчей), ошибки: mean={errors.mean():.2e}, аномалий (превышение порога батчем): {res}")

    print('\nАномальный файл:')
    segments = pipeline.process_file(FAULT_FILE_PATH)
    errors, res = detector.detect_batch(segments)
    print(f"Обработано {len(errors)} окон (батчей), ошибки: mean={errors.mean():.2e}, аномалий (превышение порога батчем): {res}")