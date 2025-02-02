import pygame, sys, random, json, copy, time

# -------------------------------
# Configuración básica (sin visualización)
# -------------------------------
pygame.init()
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 4
CELL_SIZE = WIDTH // GRID_SIZE

# Colores (para compatibilidad en caso de usar interfaz)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)

# -------------------------------
# Parámetros de Q-learning
# -------------------------------
Q = {}                # Tabla Q para Q-learning
Q_FILE = "q_table.json"
alpha = 0.7         # Tasa de aprendizaje
gamma = 0.99        # Factor de descuento
epsilon = 1.0       # Probabilidad de exploración
epsilon_decay = 0.999  # Decaimiento de epsilon por partida
epsilon_min = 0.1       # Valor mínimo de epsilon

# Roles:
# - Jugador 'B' (piezas negras): agente Q-learning
# - Jugador 'R' (piezas rojas): agente minimax
current_turn = 'B'

# -------------------------------
# Funciones comunes del juego
# -------------------------------
def initialize_board():
    """Inicializa el tablero con piezas en posiciones predeterminadas."""
    board = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    board[0][0] = 'R'
    board[0][2] = 'R'
    board[3][1] = 'B'
    board[3][3] = 'B'
    return board

def generate_moves(board, player):
    """
    Genera movimientos simples y de captura para la pieza 'player'.
    Se consideran movimientos diagonales en las 4 direcciones.
    """
    moves = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if board[row][col] == player:
                # Movimientos simples en las 4 diagonales
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE:
                        if board[new_row][new_col] is None:
                            moves.append(((row, col), (new_row, new_col)))
                # Movimientos de salto (captura)
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    jump_row, jump_col = row + 2 * dr, col + 2 * dc
                    mid_row, mid_col = row + dr, col + dc
                    if (0 <= jump_row < GRID_SIZE and 0 <= jump_col < GRID_SIZE and
                        board[jump_row][jump_col] is None and 
                        board[mid_row][mid_col] is not None and 
                        board[mid_row][mid_col] != player):
                        moves.append(((row, col), (jump_row, jump_col)))
    return moves

def handle_move(board, move):
    """Ejecuta el movimiento 'move' y retorna el nuevo estado del tablero."""
    new_board = copy.deepcopy(board)
    (old_row, old_col), (new_row, new_col) = move
    piece = new_board[old_row][old_col]
    new_board[old_row][old_col] = None
    new_board[new_row][new_col] = piece
    # Si el movimiento es de salto (captura), elimina la pieza capturada.
    if abs(new_row - old_row) == 2:
        captured_row, captured_col = (old_row + new_row) // 2, (old_col + new_col) // 2
        new_board[captured_row][captured_col] = None
    return new_board

def is_game_over(board):
    """Verifica si el juego terminó. Se considera terminado si un jugador ya no tiene piezas."""
    red_count = sum(row.count('R') for row in board)
    black_count = sum(row.count('B') for row in board)
    if red_count == 0:
        return True, "Agente Q-learning (B) gana"  # Si no quedan rojas, gana B (Q-learning)
    elif black_count == 0:
        return True, "Minimax (R) gana"           # Si no quedan negras, gana R (minimax)
    return False, None

# -------------------------------
# Funciones para Q-learning
# -------------------------------
def board_to_state(board):
    """Convierte el tablero a una cadena única para usarla como estado."""
    return str(board)

def select_action(state, moves):
    """Selecciona una acción usando la política epsilon-greedy."""
    if random.uniform(0, 1) < epsilon:
        return random.choice(moves)
    q_values = [Q.get((state, move), 0) for move in moves]
    max_q = max(q_values)
    best_moves = [m for i, m in enumerate(moves) if q_values[i] == max_q]
    return random.choice(best_moves)

# Evaluar el estado del tablero (sistema de recompensas y castigos mejorado)
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

def update_q(state, action, reward, next_state, next_moves, visited_states):
    """Actualiza la Q-table usando la fórmula del Q-learning, con penalización si el estado se repite."""
    max_future_q = max([Q.get((next_state, m), 0) for m in next_moves], default=0)
    current_q = Q.get((state, action), 0)
    repeat_penalty = -5 if next_state in visited_states else 0
    Q[(state, action)] = current_q + alpha * (reward + gamma * max_future_q + repeat_penalty - current_q)
    visited_states.add(next_state)

def move_qlearning(visited_states, board):
    """Realiza un movimiento para el agente Q-learning (piezas 'B')."""
    state = board_to_state(board)
    moves = generate_moves(board, 'B')
    if moves:
        # Priorizar movimientos de captura
        capture_moves = [m for m in moves if abs(m[0][0] - m[1][0]) == 2]
        chosen_moves = capture_moves if capture_moves else moves
        action = select_action(state, chosen_moves)
        new_board = handle_move(board, action)
        next_state = board_to_state(new_board)
        next_moves = generate_moves(new_board, 'R')
        reward = evaluate_state(new_board, last_move=action)
        update_q(state, action, reward, next_state, next_moves, visited_states)
        return new_board
    return board

# -------------------------------
# Funciones para Minimax (con poda alfa-beta)
# -------------------------------
def evaluate_board(board):
    """
    Función de evaluación para minimax.
    Retorna (black_count - red_count) + (black_moves - red_moves).
    Dado que los valores altos favorecen a 'B', el agente minimax (jugador 'R') buscará minimizar.
    """
    red_count = sum(row.count('R') for row in board)
    black_count = sum(row.count('B') for row in board)
    red_moves = len(generate_moves(board, 'R'))
    black_moves = len(generate_moves(board, 'B'))
    return (black_count - red_count) + (black_moves - red_moves)

# Tabla de transposición para evitar recalcular estados
transposition_table = {}

def minimax(board, depth, alpha_val, beta_val, maximizing_player):
    key = board_to_state(board) + f"_{depth}_{maximizing_player}"
    if key in transposition_table:
        return transposition_table[key]
    
    game_over, _ = is_game_over(board)
    if depth == 0 or game_over:
        val = evaluate_board(board)
        return val, None

    best_move = None
    # Genera y ordena movimientos; si es captura, esos movimientos se pondrán primero.
    moves = generate_moves(board, 'R')  # Usamos 'R' porque el minimax controla al jugador 'R'
    # Ordenar los movimientos: para maximizing ordenamos en forma descendente, para minimizing en ascendente.
    if maximizing_player:
        moves = sorted(moves, key=lambda m: evaluate_board(handle_move(board, m)), reverse=True)
    else:
        moves = sorted(moves, key=lambda m: evaluate_board(handle_move(board, m)))

    if maximizing_player:
        max_eval = float('-inf')
        for move in moves:
            new_board = handle_move(board, move)
            eval_val, _ = minimax(new_board, depth - 1, alpha_val, beta_val, False)
            if eval_val > max_eval:
                max_eval = eval_val
                best_move = move
            alpha_val = max(alpha_val, eval_val)
            if beta_val <= alpha_val:
                break  # Poda alfa-beta
        transposition_table[key] = (max_eval, best_move)
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in moves:
            new_board = handle_move(board, move)
            eval_val, _ = minimax(new_board, depth - 1, alpha_val, beta_val, True)
            if eval_val < min_eval:
                min_eval = eval_val
                best_move = move
            beta_val = min(beta_val, eval_val)
            if beta_val <= alpha_val:
                break  # Poda alfa-beta
        transposition_table[key] = (min_eval, best_move)
        return min_eval, best_move
    
def iterative_deepening_minimax(board, max_depth):
    """
    Realiza búsqueda iterativa hasta max_depth, devolviendo el mejor movimiento encontrado.
    """
    best_move = None
    for depth in range(1, max_depth + 1):
        # Reiniciar la tabla de transposición en cada iteración puede ser una opción si la memoria es limitada,
        # o se puede mantener para aprovechar resultados previos.
        global transposition_table
        transposition_table = {}  
        eval_val, move = minimax(board, depth, float('-inf'), float('inf'), False)
        if move is not None:
            best_move = move
        # Aquí se podría incluir un control de tiempo si fuese necesario.
    return best_move

def move_minimax(board):
    """
    Realiza un movimiento para el agente minimax (jugador 'R') utilizando búsqueda iterativa con una profundidad máxima.
    """
    max_depth = 2 
    best_move = iterative_deepening_minimax(board, max_depth)
    if best_move:
        return handle_move(board, best_move)
    return board

# -------------------------------
# Persistencia de la Q-table
# -------------------------------
def save_q_table():
    try:
        with open(Q_FILE, "w") as f:
            json.dump({str(k): v for k, v in Q.items()}, f)
    except Exception as e:
        print("Error guardando Q-table:", e)

# -------------------------------
# Ciclo de entrenamiento sin interfaz gráfica
# -------------------------------
def training_loop():
    global epsilon, current_turn
    board = initialize_board()
    current_turn = 'B'  # 'B' (Q-learning) vs 'R' (Minimax)
    visited_states = set()
    wins = {'B': 0, 'R': 0}
    losses = {'B': 0, 'R': 0}
    max_episodes = 100
    episode = 0

    while episode < max_episodes:
        # Según el turno, se ejecuta el movimiento del agente correspondiente
        if current_turn == 'B':
            board = move_qlearning(visited_states, board)
            current_turn = 'R'
        else:
            board = move_minimax(board)
            current_turn = 'B'

        # Verificar si el juego ha terminado
        over, winner_msg = is_game_over(board)
        if over:
            if "B" in winner_msg:
                wins['B'] += 1
                losses['R'] += 1
            else:
                wins['R'] += 1
                losses['B'] += 1

            episode += 1
            print(f"Episodio {episode}: {winner_msg} | Wins: {wins} | Losses: {losses} | epsilon: {epsilon:.3f}")
            board = initialize_board()
            # Alternar quién empieza según el episodio
            current_turn = 'B' if episode % 2 == 0 else 'R'
            visited_states.clear()
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

    save_q_table()
    print("Entrenamiento finalizado.")
    print(f"Resultados finales: Wins: {wins} | Losses: {losses}")

if __name__ == "__main__":
    training_loop()
