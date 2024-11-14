import os
import cv2

# Buat folder untuk menyimpan hasil snapshot
snapshot_folder = 'snapshots2'
if not os.path.exists(snapshot_folder):
    os.makedirs(snapshot_folder)

# Load Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')

# Buka kamera
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Gagal mengambil frame")
        break

    # Ubah ke grayscale untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Jika wajah terdeteksi
    for (x, y, w, h) in faces:
        # Gambar kotak kuning di sekitar wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Tekan 's' untuk menyimpan gambar
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # 1. Simpan gambar original
            original_filename = os.path.join(snapshot_folder, 'original.png')
            cv2.imwrite(original_filename, frame)
            print(f"Gambar original disimpan sebagai {original_filename}")

            # 2. Simpan gambar grayscale
            grayscale_filename = os.path.join(snapshot_folder, 'grayscale.png')
            cv2.imwrite(grayscale_filename, gray)
            print(f"Gambar grayscale disimpan sebagai {grayscale_filename}")

            # 3. Simpan gambar blackwhite (thresholding)
            _, blackwhite = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            blackwhite_filename = os.path.join(snapshot_folder, 'blackwhite.png')
            cv2.imwrite(blackwhite_filename, blackwhite)
            print(f"Gambar blackwhite disimpan sebagai {blackwhite_filename}")

            # 4. Simpan gambar yang dicrop sesuai dengan kotak kuning
            cropped_face = frame[y:y + h, x:x + w]  # Crop sesuai area wajah dalam kotak kuning
            cropped_filename = os.path.join(snapshot_folder, 'cropped_face.png')
            cv2.imwrite(cropped_filename, cropped_face)
            print(f"Gambar cropped wajah disimpan sebagai {cropped_filename}")

            # Hentikan loop setelah menyimpan gambar
            break

    # Tampilkan output
    cv2.imshow("Face Detection", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan kamera dan tutup semua jendela
cam.release()
cv2.destroyAllWindows()
