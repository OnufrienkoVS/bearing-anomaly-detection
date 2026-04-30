# opc_server.py
import time
import numpy as np
from opcua import Server
import argparse
from pathlib import Path
from datetime import datetime

class OPCUAServer:
    """Простой OPC UA сервер для передачи вибрационного сигнала"""
    
    def __init__(self, endpoint="opc.tcp://localhost:4840", fs=20000):
        self.endpoint = endpoint
        self.fs = fs
        self.server = None
        self.running = False
        
    def start(self):
        """Запуск сервера"""
        self.server = Server()
        self.server.set_endpoint(self.endpoint)
        self.server.set_server_name("Vibration Server")
        
        # Пространство имён
        uri = "http://vibration.server"
        idx = self.server.register_namespace(uri)
        
        # Создаём объекты
        objects = self.server.get_objects_node()
        vibration_obj = objects.add_object(idx, "Vibration")
        
        # Основной тег - массив сигнала
        self.signal_array = vibration_obj.add_variable(idx, "SignalArray", [0.0])
        self.signal_array.set_writable(True)
        
        # Метка времени пакета
        self.timestamp = vibration_obj.add_variable(idx, "Timestamp", "")
        self.timestamp.set_writable(True)

        # Частота дискретизации
        self.sample_rate = vibration_obj.add_variable(idx, "SampleRate", float(self.fs))
        self.sample_rate.set_writable(True)
        
        # Флаг аномалии (будет заполняться клиентом)
        self.anomaly_flag = vibration_obj.add_variable(idx, "AnomalyFlag", False)
        self.anomaly_flag.set_writable(True)
        
        # Ошибка реконструкции
        self.reconstruction_error = vibration_obj.add_variable(idx, "ReconstructionError", 0.0)
        self.reconstruction_error.set_writable(True)
        
        self.server.start()
        self.running = True
        print(f"OPC UA сервер запущен на {self.endpoint}")
        print(f"Частота дискретизации: {self.fs} Гц")
        return True
    
    def stop(self):
        """Остановка сервера"""
        self.running = False
        if self.server:
            self.server.stop()
            print("Сервер остановлен")
    
    def send_signal(self, signal_array):
        """Отправка массива сигнала"""
        if self.running and self.signal_array:
            self.signal_array.set_value(signal_array.tolist())
            self.timestamp.set_value(datetime.now().isoformat())
            return True
        return False


class SignalSimulator:
    """Симулятор подачи сигнала"""
    
    def __init__(self, file_path, fs=20000, chunk_size=20000):
        self.file_path = Path(file_path)
        self.fs = fs
        self.chunk_size = chunk_size
        self.signal = None
        self.load_signal()
    
    def load_signal(self):
        """Загрузка сигнала из файла"""
        data = []
        if self.file_path.suffix == '.npy':
            self.signal = np.load(self.file_path)
        else:
            with open(self.file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(float(line.split()[0]))
                        except:
                            continue
            self.signal = np.array(data)
        print(f"Загружен сигнал: {len(self.signal)} отсчетов")
    
    def simulate(self, callback, repeat=1):
        """Симуляция потоковой передачи"""
        total_signal = np.tile(self.signal, repeat)
        total_samples = len(total_signal)
        
        print(f"\nНачало передачи ({total_samples} отсчетов, {repeat} повторений)")
        print(f"Размер чанка: {self.chunk_size} отсчетов (1 секунда)")
        
        start_time = time.time()
        sent = 0
        
        for i in range(0, total_samples, self.chunk_size):
            chunk = total_signal[i:i+self.chunk_size]
            callback(chunk)
            sent += len(chunk)
            
            # Соблюдаем реальную частоту дискретизации
            expected_time = start_time + sent / self.fs
            sleep_time = expected_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Прогресс
            progress = 100 * sent / total_samples
            elapsed = time.time() - start_time
            print(f"\rПрогресс: {progress:.1f}% | Время: {elapsed:.1f}с", end="")
        
        print(f"\nПередача завершена! Отправлено: {sent} отсчетов")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Путь к файлу с сигналом")
    parser.add_argument("--fs", type=int, default=20000, help="Частота дискретизации")
    parser.add_argument("--chunk", type=int, default=20000, help="Размер чанка (по умолчанию 20000 = 1 сек)")
    parser.add_argument("--repeat", type=int, default=1, help="Количество повторений")
    parser.add_argument("--endpoint", default="opc.tcp://localhost:4840", help="OPC UA endpoint")
    
    args = parser.parse_args()
    
    # Создаём сервер
    server = OPCUAServer(endpoint=args.endpoint, fs=args.fs)
    server.start()
    
    # Создаём симулятор
    simulator = SignalSimulator(args.file, args.fs, args.chunk)
    
    try:
        # Функция отправки данных
        def send_callback(chunk):
            server.send_signal(chunk)
        
        # Запускаем симуляцию
        simulator.simulate(send_callback, args.repeat)
        
        # Держим сервер запущенным
        print("\nСервер работает. Нажмите Ctrl+C для остановки")
        while server.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nОстановка...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()