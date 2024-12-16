import pygame
import sys

# Configuración de Pygame
pygame.init()

# Dimensiones de la ventana
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 4
CELL_SIZE = WIDTH // GRID_SIZE

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)  # Rojo para el mensaje de victoria
GREEN = (0, 255, 0)

# Crear la ventana
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Damas 4x4")

# Fuente para mostrar el texto
font = pygame.font.Font(None, 36)

# Tablero
board = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
selected_piece = None
current_turn = "IA"  # El turno comienza con la IA (jugador "max")

# Inicializar las piezas
def initialize_board():
    board[0][0] = 'R'
    board[0][2] = 'R'
    board[3][1] = 'B'
    board[3][3] = 'B'
    return board

# Dibujar el tablero
def draw_board():
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, rect)

            piece = board[row][col]
            if piece:
                # Ajuste de sombras
                shadow_color = (100, 100, 100)  # Gris oscuro para la sombra
                shadow_offset = 3  # Desplazamiento pequeño para que se vea cerca
                shadow_center = (rect.centerx + shadow_offset, rect.centery + shadow_offset)

                if piece == 'R':  # Pieza Blanca
                    pygame.draw.circle(screen, shadow_color, shadow_center, CELL_SIZE // 3)  # Sombra
                    pygame.draw.circle(screen, WHITE, rect.center, CELL_SIZE // 3)  # Pieza
                    pygame.draw.circle(screen, (200, 200, 200), rect.center, CELL_SIZE // 3, 3)  # Contorno suave
                elif piece == 'B':  # Pieza Negra
                    pygame.draw.circle(screen, shadow_color, shadow_center, CELL_SIZE // 3)  # Sombra
                    pygame.draw.circle(screen, BLACK, rect.center, CELL_SIZE // 3)  # Pieza
                    pygame.draw.circle(screen, (50, 50, 50), rect.center, CELL_SIZE // 3, 3)  # Contorno suave

    # Mostrar el turno actual
    turn_text = font.render(f"Turno: {current_turn}", True, RED if current_turn == "IA" else GREEN)
    screen.blit(turn_text, (10, 10))



# Verificar si el juego ha terminado
def is_game_over(board):
    red_count = sum(row.count('R') for row in board)
    black_count = sum(row.count('B') for row in board)
    if red_count == 0:
        return True, "Jugador Negro gana"
    elif black_count == 0:
        return True, "Jugador Blanco gana"
    return False, None

def generate_moves(board, player):
    moves = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if board[row][col] == player:
                # Movimientos simples (diagonales)
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE and board[new_row][new_col] is None:
                        moves.append(((row, col), (new_row, new_col)))

                # Saltos (capturas)
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    jump_row, jump_col = row + 2 * dr, col + 2 * dc
                    mid_row, mid_col = row + dr, col + dc
                    if (0 <= jump_row < GRID_SIZE and 0 <= jump_col < GRID_SIZE and 
                        board[jump_row][jump_col] is None and 
                        board[mid_row][mid_col] is not None and 
                        board[mid_row][mid_col] != player):
                        moves.append(((row, col), (jump_row, jump_col)))
    return moves


# Ejecutar un movimiento
def handle_move(board, move):
    temp_board = [row[:] for row in board]  # Copia profunda
    (old_row, old_col), (new_row, new_col) = move
    piece = temp_board[old_row][old_col]
    temp_board[old_row][old_col] = None
    temp_board[new_row][new_col] = piece

    # Eliminar pieza capturada
    if abs(new_row - old_row) == 2:
        captured_row, captured_col = (old_row + new_row) // 2, (old_col + new_col) // 2
        temp_board[captured_row][captured_col] = None

    return temp_board

def evaluate_board(board):
    red_count = sum(row.count('R') for row in board)
    black_count = sum(row.count('B') for row in board)

    # Considerar movimientos disponibles
    red_moves = len(generate_moves(board, 'R'))
    black_moves = len(generate_moves(board, 'B'))

    # Valorar posiciones defensivas/ofensivas
    red_position_score = sum((row * (GRID_SIZE - 1)) for row in range(GRID_SIZE) for col in range(GRID_SIZE) if board[row][col] == 'R')
    black_position_score = sum(((GRID_SIZE - 1 - row) * (GRID_SIZE - 1)) for row in range(GRID_SIZE) for col in range(GRID_SIZE) if board[row][col] == 'B')

    return (black_count - red_count) + (black_moves - red_moves) + (black_position_score - red_position_score)


# Minimax con poda Alpha-Beta
def minimax(board, depth, alpha, beta, maximizing_player):
    game_over, winner = is_game_over(board)
    if depth == 0 or game_over:
        return evaluate_board(board), None

    best_move = None
    if maximizing_player:
        max_eval = float('-inf')
        for move in generate_moves(board, 'B'):
            new_board = handle_move(board, move)
            eval, _ = minimax(new_board, depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in generate_moves(board, 'R'):
            new_board = handle_move(board, move)
            eval, _ = minimax(new_board, depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

# Manejar clics del jugador
def handle_click(pos):
    global selected_piece, board, current_turn
    if current_turn != "Jugador":
        return  # Ignorar clics si no es el turno del jugador

    col, row = pos[0] // CELL_SIZE, pos[1] // CELL_SIZE
    if selected_piece is None:
        if board[row][col] == 'R':
            selected_piece = (row, col)
    else:
        move = (selected_piece, (row, col))
        if move in generate_moves(board, 'R'):
            board = handle_move(board, move)
            current_turn = "IA"
            ai_move()
        selected_piece = None

# IA hace un movimiento
def ai_move():
    global board, current_turn
    _, move = minimax(board, 3, float('-inf'), float('inf'), True)
    if move:
        board = handle_move(board, move)
    current_turn = "Jugador"

# Bucle principal del juego
def main():
    global board, current_turn
    board = initialize_board()
    clock = pygame.time.Clock()

    # Contador de 3 segundos antes de comenzar
    for i in range(3, 0, -1):
        screen.fill(BLACK)
        draw_board()
        countdown_text = font.render(str(i), True, RED)
        screen.blit(countdown_text, (WIDTH // 2 - countdown_text.get_width() // 2, HEIGHT // 2 - countdown_text.get_height() // 2))
        pygame.display.flip()
        pygame.time.wait(1000)  # Esperar 1 segundo

    ai_move()  # La IA hace la primera jugada

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                handle_click(event.pos)

        screen.fill(BLACK)
        draw_board()
        game_over, winner = is_game_over(board)
        if game_over:
            text = font.render(winner, True, RED)
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()

