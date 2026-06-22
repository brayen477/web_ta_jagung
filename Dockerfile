FROM python:3.9

WORKDIR /app

# Install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy semua file proyek
COPY . .

# Berikan izin tulis (write) ke folder agar aplikasi bisa menyimpan foto
RUN chmod -R 777 /app

# Buka port 7860 (wajib untuk Hugging Face Spaces)
EXPOSE 7860

# Jalankan aplikasi
CMD ["python", "app.py"]
