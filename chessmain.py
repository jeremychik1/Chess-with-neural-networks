'''
Game data sources:
ALL over the board https://ajedrezdata.com/databases/otb/over-the-board-database-aj-otb-000/
puzzles https://database.lichess.org/#standard_games
by player https://www.pgnmentor.com/files.html
everything https://sourceforge.net/projects/codekiddy-chess/files/Databases/Update2/
events by week https://theweekinchess.com/twic
by chess.com username https://github.com/knox-ber/Chess.com-PGN-Downloader
OTB by player https://www.365chess.com/chess-games.php

Bucket list:
Endgame is terrible. Train a lot more endgame positions
Reinforcement learning
'''
import torch
import time
from dataset import ChessDataset
from model import ChessModel
import helper

# Parse through all the data and store it into games
now=time.time()
games = helper.parse_pgn('Carlsen.pgn')
print(f"Parsed {len(games)} games")
print("Parsing data took: " + str(time.time()-now))

# preproccessing: map the data to a matrix
now=time.time()
X,y = helper.games_to_Xy(games)
print("Mapping games to matrix took: " + str(time.time()-now))

# Create dataset & dataloader
dataset = ChessDataset(X, y)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

# Initialize model, optimizer, and loss function
device = torch.device("cuda")
model = ChessModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training
num_epochs = 100
print("Starting training...")

for epoch in range(num_epochs):
    now=time.time()
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()  # Adjust learning rate

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f} and took: {time.time()-now}")
# Save model
torch.save(model.state_dict(), "chess_model.pth")
print("Model saved!")