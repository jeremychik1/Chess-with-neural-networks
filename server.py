from flask import Flask, request, jsonify, render_template
import chess
import predict

app = Flask(__name__)

@app.route('/bot_move', methods=['POST'])
def bot_move():
    data = request.get_json()
    board = chess.Board(data['position'])
    
    # Get the bot's move
    move = predict.get_next_move(board)

    return jsonify({"move": move})

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML file

if __name__ == '__main__':
    app.run(debug=True)
