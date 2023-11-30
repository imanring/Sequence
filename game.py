import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import torch
from scipy.special import softmax
from tqdm import tqdm

class Sequence:
    def __init__(self):
        # (50,51) are wild jacks
        # (48,49) are anti-wild jacks
        self.deck = np.hstack((np.arange(-2,52),np.arange(-2,52)))

        #represent the current state of the game
        # 0 represents an empty space
        # 1 represents a space with a black token
        # -1 represents a space with a white token
        self.tokens = np.zeros((10,10))

        #randomly assign spaces on the board to cards.
        self.board = np.random.choice(self.deck[self.deck<48],len(self.deck)-8,replace=False).reshape(10,10)
        
        #draw cards for players
        np.random.shuffle(self.deck)
        self.deck_pos = len(self.deck)-1
        self.hands = [self.deck[-5:],self.deck[-10:-5]]
        self.deck_pos -= 10
        self.turn = 1

        self.kernels = [np.diag(np.ones(5)),
                        np.flip(np.diag(np.ones(5)),axis=0),
                        np.ones(5).reshape(5,-1),
                        np.ones(5).reshape(-1,5)]

    
    def play(self, play,change_turn=True):
        """
        Parameters:
            play (tuple (int,(int,int)) ): play represented by card from hand and location on the board
            player (int): which player is making the move? -1 or 1
            change_turn (bool): do we change the turn to the other player.
        """        
        #place or remove token
        if self.hands[(self.turn+1)//2][play[0]] in [48,49]:
            self.tokens[play[1][0],play[1][1]] = 0
        else:
            self.tokens[play[1][0],play[1][1]] = self.turn
        
        #draw card
        self.hands[(self.turn+1)//2][play[0]] = self.deck[self.deck_pos]
        self.deck_pos -= 1
        if change_turn:
            self.turn *= -1
    
    def show_board(self):
        """
        Plot the board
        """
        cmap = plt.get_cmap('RdBu', 3)
        plt.matshow(self.tokens,cmap=cmap,vmin=-1,vmax=1)
        plt.colorbar(ticks=np.arange(-1,2))
        ticks = np.arange(11)-0.5
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.grid()
        plt.show()

    
    def get_moves(self):
        moves = list()
        j = 0
        for card in self.hands[(self.turn+1)//2]:
            if card<48:
                locs = np.stack(np.where(np.logical_and(self.board==card,self.tokens==0))).T
            elif card < 50:
                locs = np.stack(np.where(self.tokens==(-1*self.turn))).T
            else:
                locs = np.stack(np.where(self.tokens==0)).T            
            for loc in locs:
                moves.append((j,loc))
            j += 1
        return moves
        
    def check_winner(self):
        """Return the winning team defined by sequence of 5 tokens in a row."""
        if self.deck_pos < 0:
            return 2
        for k in self.kernels:
            conv = convolve2d(self.tokens,k,mode='valid')
            if (conv==5).sum() > 0:
                return 1
            if (conv==-5).sum() > 0:
                return -1
        return 0
    
    def _count_seq(self,temp,p_1=0):
        """
        Count sequences of length 2 to 5.
        """
        temp[temp==-1] = -5
        counts = np.zeros(4)
        for k in self.kernels:
            conv = convolve2d(temp,k,mode='valid')
            for i in range(4):
                counts[i] += (conv==(i+1+p_1)).sum()
        return counts
    
    def score(self,base=np.e):
        bcounts = self._count_seq(self.tokens.copy())
        rcounts = self._count_seq(self.tokens.copy()*-1)
        weights = np.power(np.ones(4)*base, np.arange(4)+1)
        return (weights @ bcounts - weights @ rcounts)
    
    def try_score(self,play,**kwargs):
        old_board = self.tokens.copy()
        if self.hands[(self.turn+1)//2][play[0]] in [48,49]:
            self.tokens[play[1][0],play[1][1]] = 0
        else:
            self.tokens[play[1][0],play[1][1]] = self.turn
        s = self.score(**kwargs)
        self.tokens = old_board
        return s
    
    def sample_action(self,model,T=30):
        #Sample action via the Boltzman distribution
        moves = self.get_moves()
        if len(moves)==0:
            return -1, moves
        scores = np.zeros(len(moves))
        for i in range(len(moves)):
            x = self.prepare_input(moves,i)
            scores[i] = model(x.unsqueeze(0)).item()
        p = softmax(scores/T)
        move = np.random.choice(np.arange(len(moves)),p=p)
        return move, moves
    
    def prepare_input(self,moves,move_ind):
        old_board = self.tokens.copy()
        play = moves[move_ind]
        if self.hands[(self.turn+1)//2][play[0]] in [48,49]:
            old_board[play[1][0],play[1][1]] = 0
        else:
            old_board[play[1][0],play[1][1]] = self.turn
        bcounts = self._count_seq(old_board.copy()*self.turn)
        rcounts = self._count_seq(old_board.copy()*self.turn*-1)
        c = -1
        scores = list()
        for i in range(len(moves)):
            play = moves[i]
            if play[0] != moves[move_ind][0]:
                if self.hands[(self.turn+1)//2][play[0]] not in [48,49]:
                    old_board[play[1][0],play[1][1]] = 0
                else:
                    old_board[play[1][0],play[1][1]] = self.turn
        old_board[moves[move_ind][1][0],moves[move_ind][1][1]] -= 0.5
        pot_counts = self._count_seq(old_board.copy()*self.turn,p_1=0.5)
        oej = 0
        tej = 0
        for i in range(5):
            if i != play[0]:
                oej += self.hands[(self.turn+1)//2][i] in [48,49]
                tej += self.hands[(self.turn+1)//2][i] in [50,51]
        return torch.from_numpy(np.hstack((bcounts,rcounts,pot_counts,[oej,tej]))).float()
    
    def prepare_input_(self,moves,move_ind):
        potential_moves = np.zeros((10,10))
        for m in moves:
            if m[0]!=moves[move_ind][0]:
                if self.hands[(self.turn+1)//2][m[0]] in [48,49]:
                    potential_moves[m[1][0],m[1][1]] += 1
                else:
                    potential_moves[m[1][0],m[1][1]] += 1
        
        np_input = np.stack((potential_moves,self.tokens==self.turn,self.tokens==(self.turn*-1)))
        if self.hands[(self.turn+1)//2][moves[move_ind][0]] in [48,49]:
            np_input[2,moves[move_ind][1][0],moves[move_ind][1][1]] = 0
        else:
            np_input[1+(self.turn+1)//2,moves[move_ind][1][0],moves[move_ind][1][1]] = 1
        return torch.from_numpy(np_input).float()
        
    def show_moves(self,moves):
        #moves = [m[1] for m in moves if m[0]==card]
        cmap = plt.get_cmap('RdBu', 3)
        plt.matshow(self.tokens,cmap=cmap,vmin=-1,vmax=1)
        for i in range(len(moves)):
            is_wild = self.hands[(self.turn+1)//2][moves[i][0]] in [50,51]
            plt.text(moves[i][1][1]-.2,moves[i][1][0]+0.15-is_wild*.3,i,c='black')
        plt.colorbar(ticks=np.arange(-1,2))
        ticks = np.arange(11)-0.5
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.grid()
        plt.show()