import cv2
import numpy as np
import mediapipe as mp

# Inicializar MediaPipe Hand
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicializar la cámara
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # Crear un canvas en blanco para cada frame
    canvas = np.zeros_like(frame)

    # Definir una región de interés (ROI) más grande
    roi = frame[50:750, 50:750]
    roi_canvas = canvas[50:750, 50:750]

    # Convertir ROI a RGB para el procesamiento de MediaPipe
    image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Dibujar el rectángulo de ROI en el frame original
    cv2.rectangle(frame, (50, 50), (750, 750), (0, 255, 0), 2)
    
    # Dibujar los puntos de referencia de las manos y calcular distancias
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = hand_landmarks.landmark

            # Obtener las coordenadas de los puntos de referencia
            points = []
            for idx, landmark in enumerate(landmarks):
                x = int(landmark.x * roi.shape[1])
                y = int(landmark.y * roi.shape[0])
                points.append((x, y))

                # Cambiar el color y tamaño de los puntos de referencia según la mano
                color = (0, 255, 0) if i == 0 else (0, 0, 255)
                radius = 8
                
                # Dibujar círculos en los puntos de referencia
                cv2.circle(roi_canvas, (x, y), radius, color, -1)

            # Dibujar líneas entre puntos clave (por ejemplo, entre el pulgar e índice)
            if len(points) >= 21:
                thumb_tip = points[4]
                index_tip = points[8]
                cv2.line(roi_canvas, thumb_tip, index_tip, color, 2)

                # Calcular la distancia entre el pulgar e índice
                distance = int(np.linalg.norm(np.array(thumb_tip) - np.array(index_tip)))
                cv2.putText(roi_canvas, f"Dist: {distance}px", (50, 50 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Superponer el canvas en el frame original en tiempo real
    frame = cv2.addWeighted(frame, 1, canvas, 0.7, 0)

    # Mostrar el frame
    cv2.imshow('Handtracker', frame)
    
    # Salir del bucle con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
