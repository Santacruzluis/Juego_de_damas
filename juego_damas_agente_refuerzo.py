import pygame
import sys
import random
import json
import numpy as np

# Configuración de Pygame
pygame.init()

# Dimensiones de la ventana
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 4
CELL_SIZE = WIDTH // GRID_SIZE

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Crear la ventana
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Damas 4x4 - Q-learning")

# Fuente para texto
font = pygame.font.Font(None, 36)

# Tablero y variables del juego
board = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
selected_piece = None
current_turn = "IA"  # El turno comienza con la IA
Q = {}  # Tabla Q para el aprendizaje
Q_FILE = "q_table.json"  # Archivo para guardar la tabla Q

# Parámetros de Q-learning
alpha = 0.7 # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 1.0  # Probabilidad de exploración
epsilon_decay = 0.999  # Factor de disminución de epsilon
epsilon_min = 0.1  # Mínimo valor de epsilon

# Inicializar las piezas
def initialize_board():
    board = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    # Las fichas rojas inician en la fila 0, y las negras en la fila 3
    board[0][0] = 'R'
    board[0][2] = 'R'
    board[3][1] = 'B'
    board[3][3] = 'B'
    return board

# Función para obtener direcciones válidas según la ficha (no coronada vs. coronada)
def get_directions(piece):
    # Si la ficha es normal
    if piece == 'R':
        # Las rojas se mueven hacia abajo (filas mayores)
        return [(1, -1), (1, 1)]
    elif piece == 'B':
        # Las negras se mueven hacia arriba (filas menores)
        return [(-1, -1), (-1, 1)]
    # Si es una ficha coronada (reina), puede moverse en todas las direcciones
    elif piece in ['RK', 'BK']:
        return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    return []

# Cargar la tabla Q desde el archivo, manteniendo datos previos
def load_q_table():
    global Q
    try:
        with open(Q_FILE, "r") as f:
            Q = json.load(f)  # Cargar la tabla Q desde el archivo JSON
            Q = {tuple(eval(k)): v for k, v in Q.items()}  # Convertir las claves de cadena a tuplas
    except FileNotFoundError:
        Q = {}  # Iniciar tabla vacía si el archivo no existe

# Guardar la tabla Q en el archivo, agregando nuevos datos
def save_q_table():
    try:
        with open(Q_FILE, "r") as f:
            existing_q = json.load(f)  # Leer tabla previa si existe
            existing_q.update({str(k): v for k, v in Q.items()})  # Fusionar con nuevos datos
    except (FileNotFoundError, json.JSONDecodeError):
        existing_q = {str(k): v for k, v in Q.items()}  # Si el archivo no existe o está vacío
    with open(Q_FILE, "w") as f:
        json.dump(existing_q, f)  # Guardar tabla actualizada en el archivo JSON

# Dibujar el tablero
def draw_board():
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, rect)

            piece = board[row][col]
            if piece:
                if piece[0] == 'R':  # Tanto "R" como "RK"
                    pygame.draw.circle(screen, RED, rect.center, CELL_SIZE // 3)
                elif piece[0] == 'B':  # Tanto "B" como "BK"
                    pygame.draw.circle(screen, GREEN, rect.center, CELL_SIZE // 3)

            # Resaltar la ficha seleccionada
            if selected_piece == (row, col):
                pygame.draw.rect(screen, (20, 234, 135), rect, 3)  # Borde alrededor de la ficha seleccionada

            # Dibujar posibles movimientos
            if selected_piece and ((row, col) in [move[1] for move in generate_moves(board, 'R') if move[0] == selected_piece]):
                pygame.draw.circle(screen, (145, 25, 55), rect.center, CELL_SIZE // 6)

# Generar movimientos válidos según la ficha y sus direcciones permitidas
def generate_moves(board, player):
    moves = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            piece = board[row][col]
            # Consideramos solo fichas del jugador solicitado (ya sea normal o coronada)
            if piece and (piece == player or (player == 'R' and piece == 'RK') or (player == 'B' and piece == 'BK')):
                directions = get_directions(piece)
                # Movimientos simples
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE and board[new_row][new_col] is None:
                        moves.append(((row, col), (new_row, new_col)))
                # Movimientos de salto (captura)
                for dr, dc in directions:
                    jump_row, jump_col = row + 2 * dr, col + 2 * dc
                    mid_row, mid_col = row + dr, col + dc
                    if (0 <= jump_row < GRID_SIZE and 0 <= jump_col < GRID_SIZE and
                        board[jump_row][jump_col] is None and 
                        board[mid_row][mid_col] is not None and 
                        board[mid_row][mid_col][0] != piece[0]):  # La pieza a saltar es del oponente
                        moves.append(((row, col), (jump_row, jump_col)))
    return moves

# Ejecutar un movimiento y coronar la ficha si llega al final
def handle_move(board, move):
    (old_row, old_col), (new_row, new_col) = move
    piece = board[old_row][old_col]
    board[old_row][old_col] = None
    # Si la ficha llega al extremo opuesto, se corona
    if piece == 'R' and new_row == GRID_SIZE - 1:
        piece = 'RK'
    elif piece == 'B' and new_row == 0:
        piece = 'BK'
    board[new_row][new_col] = piece

    # Si es un salto, remover la pieza capturada
    if abs(new_row - old_row) == 2:
        captured_row, captured_col = (old_row + new_row) // 2, (old_col + new_col) // 2
        board[captured_row][captured_col] = None
    return board

# Evaluar el estado del tablero (sistema de recompensas y castigos)
def evaluate_state(board, prev_board=None, last_move=None):

    # Contar piezas
    red_count = sum(row.count('R') + row.count('RK') for row in board)
    black_count = sum(row.count('B') + row.count('BK') for row in board)
    
    # Condiciones terminales
    if red_count == 0:
        return 100   # Victoria para 'B'
    elif black_count == 0:
        return -100  # Victoria para 'R'
    
    reward = 0
    # Diferencia en número de fichas
    reward += (black_count - red_count) * 2
    
    # Diferencia en reinas
    curr_black_queens = sum(row.count('BK') for row in board)
    player_queens = sum(row.count('RK') for row in board)
    reward += (curr_black_queens - player_queens) * 4

    # Si se proporciona prev_board, castigar pérdida de piezas
    if prev_board is not None:
        prev_black_count = sum(row.count('B') + row.count('BK') for row in prev_board)
        prev_black_queens = sum(row.count('BK') for row in prev_board)
        if black_count < prev_black_count:
            lost = prev_black_count - black_count
            reward -= lost * 15   # Penalización por pérdida de ficha
        if curr_black_queens < prev_black_queens:
            lost_q = prev_black_queens - curr_black_queens
            reward -= lost_q * 25  # Penalización por pérdida de reina

    # Evaluar el movimiento ejecutado
    if last_move is not None:
        (old_row, old_col), (new_row, new_col) = last_move
        # Si es un movimiento de captura, bonus
        if abs(new_row - old_row) == 2:
            reward += 10
        else:
            if new_row < old_row:
                reward += 5
            elif new_row > old_row:
                reward -= 5

    # Incentivar posiciones centrales para fichas 'B'
    center_r, center_c = GRID_SIZE/2 - 0.5, GRID_SIZE/2 - 0.5
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board[r][c] == 'B':
                dist = abs(r - center_r) + abs(c - center_c)
                reward += max(0, 5 - dist)

    # Penalizar por no estar cerca de piezas enemigas: para cada ficha 'B' sin enemigo adyacente
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board[r][c] == 'B':
                enemy_adjacent = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
                            if board[rr][cc] in ['R', 'RK']:
                                enemy_adjacent = True
                if not enemy_adjacent:
                    reward -= 4

    # Castigo: si el enemigo tiene movimientos de captura disponibles, penalizar
    enemy_moves = generate_moves(board, 'R')
    for move in enemy_moves:
        if abs(move[0][0] - move[1][0]) == 2:
            reward -= 8

    # Castigo: penalizar baja movilidad de 'B'
    ai_moves = len(generate_moves(board, 'B'))
    if ai_moves < 2:
        reward -= 10

    # Castigo extra: si hay oportunidades de captura disponibles y el movimiento no fue captura
    available_moves = generate_moves(board, 'B')
    capture_moves = [m for m in available_moves if abs(m[0][0] - m[1][0]) == 2]
    if capture_moves and last_move is not None:
        (old_row, old_col), (new_row, new_col) = last_move
        if abs(new_row - old_row) < 2:  # No se aprovechó la oportunidad de captura
            reward -= 5

    # Castigo extra: penalizar fichas 'B' en el borde sin apoyo
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board[r][c] == 'B':
                if r == 0 or r == GRID_SIZE - 1 or c == 0 or c == GRID_SIZE - 1:
                    has_support = False
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
                                if board[rr][cc] == 'B' and (dr != 0 or dc != 0):
                                    has_support = True
                    if not has_support:
                        reward -= 5

    # Castigo o bonus por la coronación:
    # Si una ficha 'B' alcanza la fila de promoción (fila 0) y no es reina, penalizar.
    for c in range(GRID_SIZE):
        if board[0][c] == 'B':
            reward -= 5
    # Bonus por tener fichas coronadas (reinas)
    if curr_black_queens > 0:
        reward += curr_black_queens * 8

    return reward

# Seleccionar acción basada en Q-learning
def select_action(state, moves):
    if random.uniform(0, 1) < epsilon:
        return random.choice(moves)
    q_values = [Q.get((state, move), 0) for move in moves]
    max_q_value = max(q_values)
    best_moves = [move for i, move in enumerate(moves) if q_values[i] == max_q_value]
    if len(best_moves) == len(moves):
        return random.choice(moves)
    return random.choice(best_moves)

visited_states = set()  # Registrar estados visitados

# Actualizar la tabla Q
def update_q(state, action, reward, next_state, next_moves):
    global visited_states
    max_future_q = max([Q.get((next_state, move), 0) for move in next_moves], default=0)
    current_q = Q.get((state, action), 0)
    repeat_penalty = -5 if next_state in visited_states else 0
    Q[(state, action)] = current_q + alpha * (reward + gamma * max_future_q + repeat_penalty - current_q)
    visited_states.add(next_state)

# Representar el estado como cadena única
def board_to_state(board):
    return json.dumps(board)

# Verificar si el juego ha terminado
def is_game_over(board):
    red_count = sum(row.count('R') + row.count('RK') for row in board)
    black_count = sum(row.count('B') + row.count('BK') for row in board)
    if red_count == 0:
        return "IA"
    elif black_count == 0:
        return "Jugador"
    return None

# IA realiza un movimiento
def ai_move():
    global board, current_turn, epsilon
    state = board_to_state(board)
    normal_moves = generate_moves(board, 'B')
    capture_moves = [move for move in normal_moves if abs(move[0][0] - move[1][0]) == 2]
    moves = capture_moves if capture_moves else normal_moves
    if moves:
        action = select_action(state, moves)
        old_board = board  # Para poder extraer información del movimiento
        board = handle_move(board, action)
        next_state = board_to_state(board)
        # Se obtiene la recompensa pasando el movimiento realizado
        reward = evaluate_state(board, last_move=action)
        next_moves = generate_moves(board, 'R')
        update_q(state, action, reward, next_state, next_moves)
    current_turn = "Jugador"
    epsilon = max(epsilon * epsilon_decay, epsilon_min)


# Jugador realiza un movimiento
def player_move(move):
    global board, current_turn
    state = board_to_state(board)
    board = handle_move(board, move)
    next_state = board_to_state(board)
    # Se invierte la recompensa para el jugador (por ejemplo)
    reward = -evaluate_state(board, last_move=move)
    next_moves = generate_moves(board, 'B')
    update_q(state, move, reward, next_state, next_moves)
    current_turn = "IA"


# Bucle principal del juego
def main():
    global board, current_turn, selected_piece  
    board = initialize_board()
    current_turn = "IA"
    selected_piece = None
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_q_table()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and current_turn == "Jugador":
                pos = event.pos
                row, col = pos[1] // CELL_SIZE, pos[0] // CELL_SIZE
                if selected_piece:
                    move = (selected_piece, (row, col))
                    if move in generate_moves(board, 'R'):
                        player_move(move)
                        selected_piece = None
                    elif board[row][col] in ['R', 'RK']:
                        selected_piece = (row, col)
                    else:
                        selected_piece = None
                elif board[row][col] in ['R', 'RK']:
                    selected_piece = (row, col)
        if current_turn == "IA" and not is_game_over(board):
            ai_move()
        winner = is_game_over(board)
        if winner:
            print(f"¡Juego terminado! Ganador: {winner}")
            save_q_table()
            pygame.quit()
            sys.exit()
        screen.fill(BLACK)
        draw_board()
        pygame.display.flip()
        clock.tick(60)
        
if __name__ == "__main__":
    main()
