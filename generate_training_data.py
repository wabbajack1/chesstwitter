#!/usr/bin/env python3
import os
import chess.pgn
import numpy as np


"""
    Before we can feed data in the net, we must first:
    1. parse chess games
    2. serialize the game in the format for the net (each move)
    3. train the net with the serialized data
"""

def make_dataset():
    x, y = [], [] # x train, y label
    values = {'1/2-1/2':0, '0-1':-1, '1-0':1} # in centipawns, maxplayer is white
    gm = 0 # number of game

    for f in os.listdir("data"):
        ff = open(os.path.join("data", f), "r")
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
                #print(board, "\n")
            y.append(value)
            gm += 1
            print("Game {:d}".format(gm))
    return 


if __name__ == "__main__":
    make_dataset()

