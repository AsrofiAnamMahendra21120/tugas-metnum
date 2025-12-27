import numpy as np
import matplotlib.pyplot as plt

def linear_regression_manual(x, y):
    """
    Menghitung slope (m) dan intercept (c) menggunakan metode Least Squares.
    Persamaan: y = mx + c
    """
    n = len(x)
    
    # Menghitung jumlah (sum) yang dibutuhkan rumus
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)
    
    # Rumus Metode Numerik Regresi Linear (Least Squares)
    # Menghitung m (kemiringan/slope)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    
    # Menghitung c (titik potong/intercept)
    c = (sum_y - m * sum_x) / n
    
    return m, c

# --- 1. Data Sampel (Misal: Luas Tanah (m2) vs Harga Rumah (Juta Rp)) ---
x_data = np.array([60, 72, 90, 100, 120, 150, 180, 200]) # Variable Independen (x)
y_data = np.array([300, 350, 480, 500, 650, 780, 950, 1100]) # Variable Dependen (y)

# --- 2. Penerapan Metode ---
m, c = linear_regression_manual(x_data, y_data)

# --- 3. Prediksi / Demonstrasi ---
# Membuat garis regresi untuk visualisasi
x_line = np.linspace(min(x_data), max(x_data), 100)
y_line = m * x_line + c

# Menampilkan Hasil Perhitungan
print("=== HASIL PERHITUNGAN METODE REGRESI LINEAR ===")
print(f"Data X (Luas): {x_data}")
print(f"Data Y (Harga): {y_data}")
print("-" * 30)
print(f"Slope (m)     : {m:.4f}")
print(f"Intercept (c) : {c:.4f}")
print(f"Model Regresi : y = {m:.4f}x + ({c:.4f})")
print("-" * 30)

# Prediksi untuk input baru (Demonstrasi)
test_luas = 110
prediksi_harga = m * test_luas + c
print(f"Demonstrasi Prediksi: Rumah seluas {test_luas}m2 diprediksi seharga {prediksi_harga:.2f} Juta")

# --- 4. Visualisasi Grafik (Penting untuk Video) ---
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='blue', label='Data Aktual')
plt.plot(x_line, y_line, color='red', label=f'Garis Regresi (y={m:.2f}x + {c:.2f})')
plt.scatter(test_luas, prediksi_harga, color='green', s=100, marker='X', label='Titik Prediksi Demo')

plt.title('Penerapan Regresi Linear: Prediksi Harga Rumah')
plt.xlabel('Luas Tanah (m2)')
plt.ylabel('Harga Rumah (Juta Rupiah)')
plt.legend()
plt.grid(True)
plt.show()
