import torch
import chess
import helper
import random
from model import ChessModel
from dataset import ChessDataset

device = torch.device("cuda")

# Load Model
model = ChessModel().to(device)
model.load_state_dict(torch.load("chess_model.pth", map_location=device))
model.eval()

def test_accuracy(test_pgn):
    games = helper.parse_pgn(test_pgn)
    X_test, y_test = helper.games_to_Xy(games)

    test_dataset = ChessDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def evaluate_model(model, dataloader, device):
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Get model predictions
            output = model(batch_X)  # Output shape: (batch_size, 4096)
            predicted_moves = torch.argmax(output, dim=1)  # Get highest probability move index

            # Compare predicted vs. correct moves
            correct += (predicted_moves == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total * 100  # Convert to percentage
    return accuracy

# Convert current board state to tensor
def board_to_tensor(board):
    matrix = helper.board_to_matrix(board)
    tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return tensor.to(device)


def get_next_move(board):
    board_tensor = board_to_tensor(board)
    with torch.no_grad():
        output = model(board_tensor)

        probabilities = torch.softmax(output, dim=1).flatten()
        top_moves = torch.topk(probabilities, 3)
    move_indices = top_moves.indices.tolist()
    move_weights = top_moves.values.tolist()

    # Ensure only legal moves get selected
    legal_moves = list(board.legal_moves)
    legal_move_indices = [helper.move_to_index(m) for m in legal_moves]

    legal_top_moves, legal_weights = [], []

    for move_index, weight in zip(move_indices, move_weights):
        if move_index in legal_move_indices:
            legal_top_moves.append(move_index)
            legal_weights.append(weight)

    # if top 3 prredictions has no legal moves, pick randomm
    if not legal_top_moves:
        legal_top_moves = legal_move_indices
        legal_weights = [1] * len(legal_top_moves)  # Equal weight if no top move is legal

    # Normalize to weights
    legal_weights = [w / sum(legal_weights) for w in legal_weights]

    # select a move based off weights
    chosen_move_index = random.choices(legal_top_moves, weights=legal_weights, k=1)[0]
    uci = helper.index_to_move(chosen_move_index, board)
    # Convert move to SAN format
    move = chess.Move.from_uci(uci)
    return board.san(move)

if __name__ == "__main__":
    board = chess.Board()
    # Play with itself
    while True:
        predicted_move = get_next_move(board)
        print(f"Predicted Move: {predicted_move}")
        board.push_san(predicted_move)
        if board.is_game_over():
            print(board.outcome())
            break