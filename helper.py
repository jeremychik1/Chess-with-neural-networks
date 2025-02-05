import numpy as np
import chess.pgn
import torch

# Takes in pgn file and returns an array with the game data
def parse_pgn(file_path):
    with open(file_path, 'r') as f:
        games = []
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
    return games

# Converts a board object into 14x8x8 matrix
def board_to_matrix(board):
    # 12 unique pieces + move FROM + move TO x 8x8 board
    matrix = np.zeros((14,8,8))
    for position, piece in board.piece_map().items():
        row,col = divmod(position,8)
        if piece.color:
            piece.piece_type += 6
        matrix[piece.piece_type-1,row,col] = 1  # piece matrix
        matrix[12,row,col] = 1                  # FROM matrix

    for move in board.legal_moves:
        row,col = divmod(move.to_square,8)
        matrix[13,row,col] = 1                  # TO matrix

    return matrix

# Converts a move object into an int
def move_to_index(move):
    return move.from_square * 64 + move.to_square

# Converts an int into a move object
def index_to_move(move_index, board):
    from_square = move_index // 64
    to_square = move_index % 64
    move = chess.Move(from_square, to_square)
    # if move isnt in legal moves, it MUST be a promotion
    if move in board.legal_moves: return move.uci() 
    move.promotion = 5  # auto promote to queen
    return move.uci() if move in board.legal_moves else None

# Convert list of games into X and y tensors for torch to use
def games_to_Xy(games):
    num_positions = sum(len(list(game.mainline_moves())) for game in games)

    X = np.zeros((num_positions, 14, 8, 8), dtype=np.float32)
    y = np.zeros((num_positions,), dtype=np.int16)

    idx = 0
    for game in games:
        b = game.board()
        for move in game.mainline_moves():
            X[idx] = board_to_matrix(b)
            y[idx] = move_to_index(move)
            b.push(move)
            idx += 1

    return torch.tensor(X), torch.tensor(y, dtype=torch.long)