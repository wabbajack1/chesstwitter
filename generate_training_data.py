#!/usr/bin/env python3

import os, sys

sys.path.append("lib/")
import chess.pgn
import numpy as np
from state import State

def make_dataset():
    """
    Before we can feed data in the net, we must first:
    1. parse chess games
    2. serialize the game in the format for the net (each move)
    3. train the net with the serialized data (find a linear combination of all the selected features)

    Returns:
        np.array: train, labels
    """
    x, y = [], [] # x train, y label
    values = {'1/2-1/2':0, '0-1':-1, '1-0':1} # in centipawns, maxplayer is white
    gm = 0 # number of game

    for f in os.listdir("data"):
        
        with open(os.path.join("data", f), "r") as ff:
            while True:

                game = chess.pgn.read_game(ff)
                
                if game is None:
                    break

                val = game.headers['Result']
                if val not in values:
                    continue
            
                board = game.board()
                value = values[val]
                for i, move in enumerate(game.mainline_moves()):
                    board.push(move)
                    data = State(board).serialize_encoder()
                    x.append(data)
                    y.append(value)
                    #print(board, "\n")
                gm += 1
                print("Game {:d}, got {} positions".format(gm, len(x)))
    return np.array(x), np.array(y)


if __name__ == "__main__":
    x, y = make_dataset()
    np.savez("chess_nn_data.npz", x, y)
    print(x.shape, y.shape)

