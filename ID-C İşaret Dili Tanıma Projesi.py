import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import tkinter as tk
import datetime


# mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Tahmin modelinin yüklenmesi
model = load_model('mp_hand_gesture')

# Kelime sınıfları dosyasının yüklenmesi
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()

# Kamera başlangıcı
cap = cv2.VideoCapture(0)

prev_class_names = {"Left": "", "Right": ""}
results = []

while True:
    # Kameradaki her kareyi oku
    _, frame = cap.read()

    x, y, c = frame.shape

    # Kamerayı dik pozisyonda tutmak ve rgb renk dönüşümü
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # El işareti tahmini alma
    result = hands.process(framergb)

    class_names = {"Left": "", "Right": ""}

    # işlem sonrası sonuç
    if result.multi_hand_landmarks:
        for idx, handslms in enumerate(result.multi_hand_landmarks):
            landmarks = []
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Kamerada el koordinatlarının algılanması
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Hareketi tahmin etmek
            prediction = model.predict([landmarks])
            class_id = np.argmax(prediction)
            class_name = classNames[class_id]

            hand = "Left" if idx == 0 else "Right"
            class_names[hand] = class_name

            print("El " + hand + " için tahmin: " + class_name)

    # tahminin kamerada yazdırılması
    cv2.putText(frame, class_names["Left"], (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, class_names["Right"], (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 0), 2, cv2.LINE_AA)

    # Sonucu kamerada göster
    cv2.imshow("Output", frame)

    if prev_class_names != class_names:
        results.append(class_names.copy())
        prev_class_names = class_names.copy()

    if cv2.waitKey(1) == ord('q'):
        break

# Kamerayı kapat ve etkin pencereyi yok et
cap.release()
cv2.destroyAllWindows()

# Deneme
print("Algılanan Kelimeler:", results)

# Results tablosunu oluşturma
df = pd.DataFrame(results)
df.index += 1  # index 1'den başlaması için
results_table = df.to_string(index=True, header=True)


# Tkinter aray

# Tkinter arayüzün
window = tk.Tk('ID-C')
results_text = tk.Text(window, height=10, width=50)
results_text.insert(tk.END, results_table)
results_text.configure(state='disabled')
results_text.tag_configure('center', justify='center')
results_text.tag_add('center', '1.0', 'end')
results_text.pack()

# Pencereyi aç
window.mainloop()
# Sonuçları dosyaya yazdır
import datetime

# Şuanki tarih ve saati al
now = datetime.datetime.now()

# Dosya ismi olarak tarih ve saat bilgisini kullan
filename = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_results.txt"

# Dosya yazma modunda aç
with open(filename, 'w') as f:
    # Tablo bilgilerini dosyaya yaz
    f.write(results_table)

print(f"Algılanan kelimelerin listesi {filename} isimli dosyaya kaydedildi.")
