import matplotlib.pyplot as plt
import numpy as np

class Sequence:
    def __init__(self):
        # (50,51) are wild jacks
        # (48,49) are anti-wild jacks
        self.deck = np.hstack((np.arange(52),np.arange(52)))

        #represent the current state of the game
        # 0 represents an empty space
        # 1 represents a space with a black token
        # -1 represents a space with a white token
        self.tokens = np.zeros((10,10))

        #randomly assign spaces on the board to cards.
        board = np.concatenate((np.random.choice(self.deck[self.deck<48],len(self.deck)-8,replace=False),
                                [-1]*4)).reshape(10,10)
        board[9,-4:-1] = [board[0,0],board[0,9],board[9,0]]
        board[0,0],board[0,9],board[9,0] = (-1,-1,-1)
        self.board = board
        
        #draw cards for players
        np.random.shuffle(self.deck)
        self.deck_pos = len(self.deck)-1
        self.hands = [self.deck[-5:],self.deck[-10:-5]]
        self.deck_pos -= 10
        self.turn = 1
    
    def play(self, play, player):
        """
        Parameters:
            play (tuple (int,(int,int)) ): play represented by card from hand and location on the board
            player (int): which player is making the move? -1 or 1
        """
        #if legal move
        
        #place or remove token
        if self.hands[(player+1)//2][play[0]] in [48,49]:
            self.tokens[play[1][0],play[1][1]] = 0
        else:
            self.tokens[play[1][0],play[1][1]] = player
        
        #draw card
        self.hands[(player+1)//2][play[0]] = self.deck[self.deck_pos]
        self.deck_pos -= 1

        #if deck is empty
            #reshuffle
        
        #change turns
        self.turn = -1*self.turn
    
    def show_board(self):
        cmap = plt.get_cmap('RdBu', 3)
        plt.matshow(self.tokens,cmap=cmap,vmin=-1,vmax=1)
        plt.colorbar(ticks=np.arange(-1,2))
        plt.show()
    
    def get_moves(self,player):
        moves = list()
        j = 0
        for card in self.hands[(player+1)//2]:
            if card<48:
                locs = np.stack(np.where(np.logical_and(self.board==card,self.tokens==0))).T
            elif card < 50:
                locs = np.stack(np.where(self.tokens==(-1*player))).T
            else:
                locs = np.stack(np.where(self.tokens==0)).T
                #need to consider wild card locations
            
            for loc in locs:
                moves.append((j,loc))
            j += 1
        return moves
        
    #def check_winner(self):
        