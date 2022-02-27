#!/usr/bin/env python3
import sys
import chess
from torch import nn
import numpy as np


class State(object):
    def __init__(self, board):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

        self.values = {"p": 1, "b":2, "n": 3, "r": 4, "q":5, "k":6, 
                      "P": 7, "B":8, "N": 9, "R": 10, "Q":11, "K":12} # we have to encode the states (here look at FEN states)

    def current_state(self):
        return (self.board.board_fen(), self.board.ep_square, self.board.castling_rights, self.board.turn)

    def serialize(self):
        assert self.board.is_valid()
        state = np.zeros((64,))

        


if __name__ == "__main__":
    pass