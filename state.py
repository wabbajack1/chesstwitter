#!/usr/bin/env python3
import sys

sys.path.append("lib/python3.9/site-packages")

import chess
import numpy as np



class State(object):
    def __init__(self, board):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

        self.values = {"p": 2, "b":4, "n": 8, "r": 16, "q":32, "k":64, 
                      "P": 128, "B":256, "N": 512, "R": 1024, "Q":2048, "K":4096} # we have to encode the states (here look at FEN states)

    def current_state(self):
        return (self.board.board_fen(), self.board.ep_square, self.board.castling_rights, self.board.turn)

    def serialize_encoder(self):
        """
        Serialize the data to a representation, that we can feed out Model (here an CNN). So we Bitboard the states.
        --> https://arxiv.org/pdf/1711.09667.pdf deepchess ideas. CNN is incorporated into a new
        form of alpha-beta search.
        """
        assert self.board.is_valid()
        
        # additional states for represent the board
        bit_board = np.zeros((64,), dtype=np.int64)
        state = np.zeros((16, 8, 8), dtype=np.int64)

        for i in range(len(bit_board)):
            p_symbol = self.board.piece_at(i)
            if p_symbol is not None:
                bit_board[i] = self.values[p_symbol.symbol()] # insert the values

        if self.board.has_kingside_castling_rights(chess.WHITE):
            cas = np.zeros((8,8))
            cas[0,0] = 1
            state[0] = cas
        if self.board.has_queenside_castling_rights(chess.WHITE):
            cas = np.zeros((8,8))
            cas[0,7] = 1
            state[1] = cas
        if self.board.has_kingside_castling_rights(chess.BLACK):
            cas = np.zeros((8,8))
            cas[-1,0] = 1
            state[2] = cas
        if self.board.has_queenside_castling_rights(chess.BLACK):
            cas = np.zeros((8,8))
            cas[-1,7] = 1
            state[3] = cas

        bit_board = bit_board.reshape((8,8))

        bits = 1
        for i in range(4, state.shape[0]):
            state[i] = (bit_board >> bits)&1
            bits += 1
            #print(i, bits)

        return state


if __name__ == "__main__":
    board = chess.Board()
    s = State(board)
    s.serialize_encoder()