import psutil
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import ttk


# Функція для збору даних про використання ресурсів
def collect_data(duration=60, interval=1):
    data = []
    for _ in range(int(duration / interval)):
        usage = {
            'time': time.time(),
            'cpu': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent
        }
        data.append(usage)
        time.sleep(interval)
    return pd.DataFrame(data)


# Функція для завершення процесів для звільнення оперативної пам'яті
def optimize_memory(threshold=80):
    if psutil.virtual_memory().percent > threshold:
        # Отримуємо список усіх процесів
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
            try:
                # Якщо процес використовує більше 1% оперативної пам'яті, спробуємо його завершити
                if proc.info['memory_percent'] > 1:
                    psutil.Process(proc.info['pid']).terminate()
                    print(f"Terminated {proc.info['name']} (PID {proc.info['pid']}) to free memory")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue


# Завантаження даних
data = collect_data(duration=60, interval=1)
data.to_csv('resource_usage.csv', index=False)

# Підготовка даних для моделі
X = data[['cpu', 'memory']]
y = data['disk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Навчання моделі 
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# Створення графічного інтерфейсу
class ResourceMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resource Monitor")

        # Створення основної рамки
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Підписи для полів
        ttk.Label(self.main_frame, text="CPU Usage (%)").grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(self.main_frame, text="Memory Usage (%)").grid(row=1, column=0, padx=5, pady=5)
        ttk.Label(self.main_frame, text="Actual Disk Usage (%)").grid(row=2, column=0, padx=5, pady=5)
        ttk.Label(self.main_frame, text="Predicted Disk Usage (%)").grid(row=3, column=0, padx=5, pady=5)

        # Поля для виводу інформації
        self.cpu_label = ttk.Label(self.main_frame, text="0", font=("Helvetica", 12))
        self.cpu_label.grid(row=0, column=1, padx=5, pady=5)

        self.memory_label = ttk.Label(self.main_frame, text="0", font=("Helvetica", 12))
        self.memory_label.grid(row=1, column=1, padx=5, pady=5)

        self.disk_label = ttk.Label(self.main_frame, text="0", font=("Helvetica", 12))
        self.disk_label.grid(row=2, column=1, padx=5, pady=5)

        self.predicted_disk_label = ttk.Label(self.main_frame, text="0", font=("Helvetica", 12))
        self.predicted_disk_label.grid(row=3, column=1, padx=5, pady=5)

        # Починаємо оновлювати ресурси
        self.update_resources()

    def update_resources(self):
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent

        # Створюємо DataFrame з іменами колонок
        features = pd.DataFrame([[cpu, memory]], columns=['cpu', 'memory'])

        # Масштабуємо дані
        features_scaled = scaler.transform(features)

        # Передбачаємо використання диску
        predicted_disk_usage = model.predict(features_scaled)[0]

        # Оновлюємо поля з інформацією
        self.cpu_label.config(text=f"{cpu}%")
        self.memory_label.config(text=f"{memory}%")
        self.disk_label.config(text=f"{disk}%")
        self.predicted_disk_label.config(text=f"{predicted_disk_usage:.2f}%")

        # Перевіряємо використання пам'яті та викликаємо оптимізацію
        if memory > 80:
            optimize_memory(threshold=80)

        # Оновлюємо кожні 5 секунд
        self.root.after(5000, self.update_resources)


# Запуск програми
root = tk.Tk()
app = ResourceMonitorApp(root)
root.mainloop()
