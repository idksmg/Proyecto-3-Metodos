# === Proyecto: Fourier Drawing con animación y procesamiento de imagen ===

import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.fft import fft, fftfreq
from matplotlib.animation import FuncAnimation
import os

# === 1. VALIDAR DEPENDENCIAS ===
try:
    import cv2
    import numpy
    import matplotlib
except ImportError as e:
    raise ImportError("Faltan dependencias. Instala 'numpy', 'matplotlib' y 'opencv-python' con: pip install numpy matplotlib opencv-python")

# === 2. EXTRAER CONTORNO DESDE IMAGEN ===
def extract_contour_from_image(image_path, num_points=500, show_preview=False):
    """Extrae el contorno de una imagen en formato PNG, JPG o JPEG usando detección de bordes."""
    # Validar formato de archivo
    valid_extensions = ['.png', '.jpg', '.jpeg']
    ext = os.path.splitext(image_path.lower())[1]
    if ext not in valid_extensions:
        raise ValueError(f"Formato no soportado: {ext}. Usa imágenes en formato PNG, JPG o JPEG.")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen en: {image_path}")
    
    # Cargar imagen
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}. Verifica la ruta o el formato.")
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque para reducir ruido 
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detección de bordes con Canny
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        raise ValueError("No se encontraron contornos. Usa una imagen con bordes claros o ajusta los parámetros de Canny (threshold1, threshold2).")
    
    # Seleccionar el contorno más grande
    largest = max(contours, key=cv2.contourArea)[:, 0, :]
    if len(largest) < num_points:
        print(f"Advertencia: El contorno tiene solo {len(largest)} puntos. Reduciendo num_points a {len(largest)}.")
        num_points = len(largest)
    
    # Muestrear puntos
    step = max(1, len(largest) // num_points)
    sampled = largest[::step][:num_points]
    
    # Mostrar previsualización si se solicita
    if show_preview:
        preview = img.copy()
        cv2.drawContours(preview, [largest], -1, (0, 255, 0), 2)
        # Mostrar también los bordes detectados por Canny
        cv2.imshow("Contorno Detectado (Verde) y Bordes Canny (Ventana Separada)", preview)
        cv2.imshow("Bordes Canny", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Convertir a coordenadas complejas y centrar
    x = sampled[:, 0] - np.mean(sampled[:, 0])
    y = -sampled[:, 1] + np.mean(sampled[:, 1])
    
    return x + 1j * y, np.max(np.abs(x)), np.max(np.abs(y))

# === 3. OBTENER TRANSFORMADA DE FOURIER ===
def compute_fourier_coefficients(z):
    """Calcula los coeficientes de Fourier y las frecuencias."""
    N = len(z)
    Z = fft(z) / N
    freqs = fftfreq(N, d=1/N)
    indices = np.argsort(-np.abs(Z))  # Ordenar por magnitud
    return Z[indices], freqs[indices]

# === 4. ANIMACIÓN CON EPICICLOS ===
def animate_epicycles(coeffs, freqs, num_vectors=300, x_max=300, y_max=300, save_animation=False, output_path="fourier_animation.mp4"):
    """Crea una animación de epiciclos basada en los coeficientes de Fourier."""
    N = len(coeffs)
    num_vectors = min(num_vectors, N)
    t_vals = np.linspace(0, 1, N, endpoint=False)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-x_max * 1.2, x_max * 1.2)
    ax.set_ylim(-y_max * 1.2, y_max * 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    path, = ax.plot([], [], color='purple', lw=2, label='Figura reconstruida')
    epicycle_lines, = ax.plot([], [], color='gray', lw=1, label='Epiciclos')
    ax.legend(loc='upper right')
    
    points = []
    
    def init():
        path.set_data([], [])
        epicycle_lines.set_data([], [])
        return path, epicycle_lines
    
    def update(frame):
        t = t_vals[frame % N]
        x, y = 0.0, 0.0
        segments_x = []
        segments_y = []
        
        for n in range(num_vectors):
            freq = freqs[n]
            coeff = coeffs[n]
            prev_x, prev_y = x, y
            angle = 2 * np.pi * freq * t
            x += np.real(coeff * np.exp(1j * angle))
            y += np.imag(coeff * np.exp(1j * angle))
            segments_x.extend([prev_x, x, None])
            segments_y.extend([prev_y, y, None])
        
        points.append((x, y))
        if len(points) > N:
            points.pop(0)
        
        px, py = zip(*points)
        path.set_data(px, py)
        epicycle_lines.set_data(segments_x, segments_y)
        
        return path, epicycle_lines
    
    anim = FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval=20)
    
    if save_animation:
        try:
            anim.save(output_path, writer='ffmpeg', fps=30)
            print(f"Animación guardada en: {output_path}")
        except Exception as e:
            print(f"Error al guardar la animación: {e}. Asegúrate de tener 'ffmpeg' instalado.")
    
    plt.show()
    return anim

# === 5. EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    image_path = input("Ingresa la ruta de la imagen (PNG, JPG, JPEG, ej: assets/corazon.jpg)" ) or os.path.join("assets", "ejemplo.jpg")
    
    try:
        # Extraer contorno con previsualización
        z, x_max, y_max = extract_contour_from_image(image_path, num_points=300, show_preview=True)
        # Calcular coeficientes de Fourier
        coeffs, freqs = compute_fourier_coefficients(z)
        # Crear animación
        animate_epicycles(coeffs, freqs, num_vectors=700, x_max=x_max, y_max=y_max, save_animation=True)
    except Exception as e:
        print(f"Error: {e}")
        print("Sugerencias:")
        print("- Asegúrate de que la imagen sea PNG, JPG o JPEG con bordes claros.")
        print("- Para imágenes JPEG con compresión, ajusta los parámetros de Canny (threshold1, threshold2) si no se detectan contornos.")
        print("- Instala las dependencias con: pip install numpy matplotlib opencv-python")




      
