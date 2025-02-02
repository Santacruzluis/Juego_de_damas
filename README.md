Este repositorio contiene dos implementaciones relacionadas con el juego de damas 4x4:

Entrenamiento de la IA (Q-learning vs. Minimax):
En este archivo, la IA se entrena mediante auto‑juego (self‑play) enfrentando a dos agentes:

El agente Q-learning (piezas negras, representado por 'B'), que aprende a partir de la experiencia y guarda una tabla Q en formato JSON.
El agente Minimax (piezas rojas, representado por 'R'), que utiliza el algoritmo Minimax con poda alfa‑beta, ordenamiento de movimientos y tabla de transposición para tomar decisiones de forma competitiva.
Juego IA vs. Humano:
En este archivo se implementa la posibilidad de jugar contra la IA entrenada. Si ya se cuenta con la tabla Q guardada en q_table.json, la IA utilizará lo aprendido para enfrentar al jugador humano a través de una interfaz gráfica desarrollada con Pygame.

\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

Requisitos
Para ejecutar los scripts es necesario contar con:

Python 3.13.0

Pygame
Instálalo con:
pip install pygame

Numpy (utilizado en el script de IA vs. Humano)
Instálalo con:
pip install numpy

JSON
(El módulo json viene incluido en Python)

\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

Archivos del Proyecto

entrenamiento_ia.py

Script para entrenar la IA mediante auto‑juego (Q-learning vs. Minimax).
Se ejecuta en modo sin interfaz gráfica, mostrando estadísticas de victorias y derrotas en consola.
La tabla Q se guarda en q_table.json al finalizar el entrenamiento.

juego_damas_agente_refuerzo.py

Script para enfrentar a la IA entrenada contra un jugador humano.
Se crea una ventana de Pygame donde el jugador puede seleccionar y mover sus piezas.
La IA utiliza la tabla Q previamente entrenada para tomar decisiones durante la partida.

\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

Cómo Ejecutar
Entrenamiento de la IA
Si deseas entrenar (o continuar el entrenamiento) de la IA, ejecuta el siguiente comando:
python entrenamiento_ia.py

El script realizará 10,000 episodios de auto‑juego, en los cuales se enfrentan el agente Q-learning (piezas negras) y el agente Minimax (piezas rojas). Se imprimirán las estadísticas de cada episodio (victorias, derrotas y el valor actual de ε). Al finalizar, se guarda la tabla Q en q_table.json.

Juego: IA vs. Humano
Si ya cuentas con la tabla Q entrenada (o decides entrenar previamente) y deseas jugar contra la IA, ejecuta:

python juego_damas_agente_refuerzo.py

La ventana gráfica mostrará el tablero de damas 4x4.
El jugador humano controla las piezas rojas ('R'), y la IA (con las piezas negras 'B') se moverá utilizando la tabla Q.
Para realizar un movimiento, selecciona una pieza con el ratón y, a continuación, elige la casilla destino.
La IA se encargará de responder automáticamente.

\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

Contexto del Proyecto
Este proyecto tiene como objetivo explorar la aplicación de técnicas de aprendizaje por refuerzo en un entorno de juego sencillo. La IA Q-learning se entrena mediante auto‑juego enfrentándose a un agente clásico basado en Minimax con poda alfa‑beta, lo que permite:

Acumular experiencia: La tabla Q se guarda en un archivo JSON para que, en futuras sesiones, la IA inicie con conocimientos previos y mejore su desempeño.
Experimentar con estrategias híbridas: La combinación de Q-learning y Minimax brinda un entorno desafiante para evaluar la efectividad del aprendizaje por refuerzo en juegos de estrategia.
Además, se ofrece la posibilidad de enfrentar a la IA entrenada contra un jugador humano, permitiendo comparar el desempeño de la IA y ver cómo evoluciona con el tiempo.
