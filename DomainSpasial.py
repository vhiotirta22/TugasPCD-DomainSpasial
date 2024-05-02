print("Nama\t: Vhio Tirta Anggun Azizah")
print("Stambuk\t: F55122066\n")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, minimum_filter, maximum_filter

# Baca gambar dari file (dalam format warna)
image_path = 'Bread.jpg'  # Ganti dengan path gambar Anda
image_color = cv2.imread(image_path)  # Baca gambar dalam format warna

# Ubah gambar berwarna menjadi grayscale
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Function untuk menampilkan beberapa gambar dalam grid
def show_images_in_grid(titles, images, rows, cols, figsize=(10, 10)):
    fig, axes = plt.subplots(rows, cols,  figsize=figsize)
    for ax, title, img in zip(axes.flatten(), titles, images):
        ax.set_title(title)
        ax.imshow(img if len(img.shape) == 3 else img, cmap='gray' if len(img.shape) == 2 else None)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Difference Image antara dua gambar
difference = cv2.absdiff(image_gray, image_gray)

# Average Filter dengan kernel 5x5
avg_filtered = cv2.blur(image_gray, (5, 5))

# Median Filter dengan kernel 5x5
median_filtered = median_filter(image_gray, size=5)

# Min Filter dengan kernel 3x3
min_filtered = minimum_filter(image_gray, size=3)

# Max Filter dengan kernel 3x3
max_filtered = maximum_filter(image_gray, size=3)

# Laplacian
laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)

# Sobel Gradient dalam arah X
sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)

# Sobel Gradient dalam arah Y
sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)

# Daftar judul dan gambar untuk ditampilkan dalam grid
titles = [
    "Gambar Asli (Berwarna)",
    "Difference Image",
    "Average Filter (5x5)",
    "Median Filter (5x5)",
    "Min Filter (3x3)",
    "Max Filter (3x3)",
    "Laplacian",
    "Sobel Gradient (X)",
    "Sobel Gradient (Y)"
]

images = [
    cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB),  # Konversi BGR ke RGB untuk plot berwarna
    difference,
    avg_filtered,
    median_filtered,
    min_filtered,
    max_filtered,
    laplacian,
    sobel_x,
    sobel_y
]

# Tampilkan gambar dalam grid 3x3
show_images_in_grid(titles, images, 3, 3, figsize=(7, 7))