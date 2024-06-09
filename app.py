import cv2              # Importa la biblioteca OpenCV para el procesamiento de imágenes y video.
import mediapipe as mp  # Importa la biblioteca MediaPipe para el seguimiento de mano.
from Funciones.condicionales import condicionalesLetras   # Importa una función personalizada para el procesamiento de gestos.
from Funciones.normalizacionCords import obtenerAngulos    # Importa una función personalizada para calcular ángulos entre puntos de referencia de la mano.

lectura_actual = 0      # Variable para almacenar la lectura actual de un punto de referencia.

# Configuración de MediaPipe para el seguimiento de mano.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# Configuración de la captura de video desde la cámara.
cap = cv2.VideoCapture(0)  # Puede cambiar el valor a una ruta de archivo de video si desea leer desde un archivo.

# Configuración de la resolución de la cámara.
wCam, hCam = 1280, 720
cap.set(3, wCam)
cap.set(4, hCam)

# Inicia el seguimiento de mano con MediaPipe.
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.75) as hands:

    while True:
        ret, frame = cap.read()  # Lee un fotograma desde la cámara.
        if ret == False:         # Si no se puede leer un fotograma, sale del bucle.
            break

        height, width, _ = frame.shape   # Obtiene las dimensiones del fotograma.
        frame = cv2.flip(frame, 1)       # Voltea horizontalmente el fotograma (espejo).
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convierte el fotograma a formato RGB (necesario para MediaPipe).
        results = hands.process(frame_rgb)  # Procesa el fotograma para detectar manos.

        if results.multi_hand_landmarks is not None:  # Si se detecta al menos una mano en el fotograma:

            # Calcula los ángulos entre puntos de referencia de la mano.
            angulosid = obtenerAngulos(results, width, height)[0]

            dedos = []  # Lista para almacenar el estado de los dedos (doblando o extendiendo).
            # Comprueba si el pulgar externo está doblado.
            if angulosid[5] > 125:
                dedos.append(1)
            else:
                dedos.append(0)
            # Comprueba si el pulgar interno está doblado.
            if angulosid[4] > 150:
                dedos.append(1)
            else:
                dedos.append(0)
            # Comprueba si los 4 dedos restantes están doblados.
            for id in range(0, 4):
                if angulosid[id] > 90:
                    dedos.append(1)
                else:
                    dedos.append(0)
                
            TotalDedos = dedos.count(1)  # Calcula el total de dedos doblados.
            condicionalesLetras(dedos, frame)  # Llama a una función para procesar los gestos detectados.

            # Calcula la posición del dedo meñique.
            pinky = obtenerAngulos(results, width, height)[1]
            pinkY = pinky[1] + pinky[0]   # Suma las coordenadas Y de los puntos de referencia del dedo meñique.
            resta = pinkY - lectura_actual  # Calcula la diferencia entre la posición actual y la anterior del dedo meñique.
            lectura_actual = pinkY  # Actualiza la posición anterior del dedo meñique.

            # Si se detecta un gesto específico, realiza una acción.
            if dedos == [0, 0, 1, 0, 0, 0]:
                if abs(resta) > 30:
                    print("jota en movimento")  # Imprime un mensaje si el gesto "J" está en movimiento.
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.rectangle(frame, (0, 0), (100, 100), (255, 255, 255), -1)  # Dibuja un rectángulo en el fotograma.
                    cv2.putText(frame, 'J', (20, 80), font, 3, (0, 0, 0), 2, cv2.LINE_AA)  # Coloca el texto "J" en el fotograma.

            # Dibuja las manos detectadas y las conexiones entre los puntos de referencia en el fotograma.
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('Frame', frame)  # Muestra el fotograma procesado en una ventana.

        if cv2.waitKey(1) & 0xFF == 27:  # Sale del bucle si se presiona la tecla Esc (código ASCII 27).
            break

cap.release()         # Libera la captura de video.
cv2.destroyAllWindows()  # Cierra todas las ventanas.
