import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_edge_detection(image_path):
    # 1. Load Citra (Grayscale)
    # Mengubah citra menjadi matriks intensitas 2D
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Gambar tidak ditemukan. Pastikan nama file 'input.jpg' benar.")
        return

    # 2. Definisi Kernel Sobel (Penerapan Turunan Numerik)
    # Kernel Gx (Beda hinga arah horizontal)
    Gx_kernel = np.array([[-1, 0, 1], 
                          [-2, 0, 2], 
                          [-1, 0, 1]])
    
    # Kernel Gy (Beda hingga arah vertikal)
    Gy_kernel = np.array([[-1, -2, -1], 
                          [0, 0, 0], 
                          [1, 2, 1]])

    # 3. Implementasi Konvolusi (Menghitung Gradien)
    # Menggunakan filter2D dari OpenCV untuk efisiensi komputasi numerik
    grad_x = cv2.filter2D(img, cv2.CV_64F, Gx_kernel)
    grad_y = cv2.filter2D(img, cv2.CV_64F, Gy_kernel)

    # 4. Menghitung Magnitudo Gradien
    # G = sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalisasi hasil ke rentang 0-255 untuk visualisasi
    magnitude = np.uint8(np.absolute(magnitude))

    # 5. Visualisasi Hasil
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Citra Asli")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Hasil Deteksi Tepi (Sobel)")
    plt.imshow(magnitude, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('hasil_demonstrasi.png') # Simpan untuk laporan
    plt.show()

# --- Jalankan Program ---
# Ganti 'input.jpg' dengan nama file gambar Anda
if __name__ == "__main__":
    sobel_edge_detection('input.jpg')