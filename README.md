# Chesstwitter
    > play with your friends through twitter chess or play against a strong AI


## Eval
- We need an eval funtion (look at the evaluation after playing all possible moves) --> exploit the number with the heighest number!
- one-ply search: "half" move --> e2e4 (Pawn to e4)
- values (weights) for material are Queen=9, Rook=5; Knight or Bishop=3; Pawn=1; King=inf (Einheit in "Pawns")
- Intention: One neural network for the entire evaluation function
    - Input: features from the Board
    - Output: one integer in centipawns
- We need a heur. eval functions __and__ the minimax Algo. for for selecting the best possible next move
- f(p) = 200(K-K')
       + 9(Q-Q')
       + 5(R-R')
       + 3(B-B' + N-N')
       + 1(P-P')
       - 0.5(D-D' + S-S' + I-I')
       + 0.1(M-M') + ...◊
    -  KQRBNP = number of kings, queens, rooks, bishops, knights and pawns
    -  D,S,I = doubled, blocked and isolated pawns
    -  M = Mobility (the number of legal moves)