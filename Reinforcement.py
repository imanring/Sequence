import random
import copy
import sys
import time

# Example card: {"number": "K", "suite": "clubs"}

"""
Globals: What are the available numbers and suites
Jacks of the given suites are excluded in this list but will be added in the deck
Spade and hearts have single eyed jacks, clubs and diamonds have two eyes
"""
# available_numbers = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
available_numbers = ["A", "2", "3", "4", "5", "6"]
# available_suites = ["hearts", "diamonds", "spades", "clubs"]
available_suites = ["hearts", "diamonds"]

def action_to_int(move, size):
    """
    Convert move (card, location) into a single dimension and return the index
    """
    card = move[0]
    loc = move[1]
    if card["number"] == "J":
        card_loc = len(available_numbers)
    else:
        card_loc = available_numbers.index(card["number"])
    int_action = loc[0] + loc[1]*size + card_loc*size*size + available_suites.index(card["suite"])*size*size*(len(available_numbers) + 1)
    return int_action

def int_to_action(idx, size):
    """
    Convert index to corresponding move (card, location)
    """
    D1 = size
    D2 = size
    D3 = len(available_numbers) + 1
    D4 = len(available_suites)
    i = idx % D1
    j = ((idx - i)//D1)%D2
    k = ( ( idx - j * D1 - i ) // (D1 * D2) ) % D3
    l = ( ( idx - k * D2 * D1 - j * D1 - i ) // (D1 * D2 * D3) ) % D4
    card = {}
    if k == len(available_numbers):
        card["number"] = "J"
    else:
        card["number"] = available_numbers[k]
    card["suite"] = available_suites[l]

    return (card, (i, j))

def dot(a, b):
    """
    Dot product utility
    """
    assert len(a) == len(b)
    c = 0
    for i in range(len(a)):
        c += a[i]*b[i]
    return c

def check_boundaries(i, j, size):
    """
    Check boundary conditions
    """
    if i >= 0 and j >= 0 and i < size and j < size:
        return True
    return False

def calculate_potentials(i, j, curr_board, color, pot):
    """
    Calculate the length of sequences already formed ending at i, j and joining squares up and left
    Drawback/To Do: This ignores potential sequences with gaps in-between that can be filled
    """
    if curr_board.board[i][j].status == color:
        if (check_boundaries(i, j - 1, curr_board.size)):
            pot[i][j][0] = pot[i][j - 1][0] + 1 # Increment potentials in rows
        else:
            pot[i][j][0] = 1
        if (check_boundaries(i - 1, j, curr_board.size)):
            pot[i][j][1] = pot[i - 1][j][1] + 1 # Increment potentials in columns
        else:
            pot[i][j][1] = 1
        if (check_boundaries(i - 1, j - 1, curr_board.size)):
            pot[i][j][2] = pot[i - 1][j - 1][2] + 1 # Increment potentials in diagonals
        else:
            pot[i][j][2] = 1

def get_features(curr_board, cards):
    """
    Feature extractor: Takes board state and cards in hand and find the features
    """
    one_hot = [0]*30
    pot_r = [[[0,0,0] for i in range(curr_board.size)] for j in range(curr_board.size)]
    pot_b = [[[0,0,0] for i in range(curr_board.size)] for j in range(curr_board.size)]
    for k in range(2 * (curr_board.size - 1) + 1):
        for i in range(curr_board.size):
            j = k - i
            if check_boundaries(i, j, curr_board.size) == False:
                pass
            else:
                calculate_potentials(i, j, curr_board, "red", pot_r)
                calculate_potentials(i, j, curr_board, "blue", pot_b)
    p1_num_seq = [0]*9
    p2_num_seq = [0]*9
    for i in range(len(pot_r)):
        for j in range(len(pot_r)):
            for k in range(3):
                p1_num_seq[pot_r[i][j][k]] += 1
    for i in range(len(pot_b)):
        for j in range(len(pot_b)):
            for k in range(3):
                p2_num_seq[pot_b[i][j][k]] += 1
    # print(p1_num_seq)
    # print(p2_num_seq)

    # Number of 5 length sequences formed for p1
    if p1_num_seq[4] <= 0:
        one_hot[0] = 1
    else:
        one_hot[1] = 1
    # Number of 5 length sequences formed for p2
    if p2_num_seq[4] <= 0:
        one_hot[2] = 1
    else:
        one_hot[3] = 1

    # Number of 4 length sequences formed for p1
    if p1_num_seq[4] <= 0:
        one_hot[4] = 1
    elif p1_num_seq[4] <= 2:
        one_hot[5] = 1
    else:
        one_hot[6] = 1
    # Number of 4 length sequences formed for p2
    if p2_num_seq[4] <= 0:
        one_hot[7] = 1
    elif p2_num_seq[4] <= 2:
        one_hot[8] = 1
    else:
        one_hot[9] = 1

     # Number of 3 length sequences formed for p1
    if p1_num_seq[3] <= 2:
        one_hot[10] = 1
    elif p1_num_seq[3] <= 5:
        one_hot[11] = 1
    else:
        one_hot[12] = 1
    # Number of 3 length sequences formed for p2
    if p2_num_seq[3] <= 2:
        one_hot[13] = 1
    elif p2_num_seq[3] <= 5:
        one_hot[14] = 1
    else:
        one_hot[15] = 1

    # Number of 2 length sequences formed for p1
    if p1_num_seq[2] <= 5:
        one_hot[16] = 1
    elif p1_num_seq[2] <= 10:
        one_hot[17] = 1
    else:
        one_hot[18] = 1
    # Number of 2 length sequences formed for p2
    if p1_num_seq[2] <= 5:
        one_hot[19] = 1
    elif p1_num_seq[2] <= 10:
        one_hot[20] = 1
    else:
        one_hot[21] = 1

    # Number of 1 length sequences formed for p1
    if p1_num_seq[1] <= 10:
        one_hot[22] = 1
    elif p1_num_seq[1] <= 20:
        one_hot[23] = 1
    else:
        one_hot[24] = 1
    # Number of 1 length sequences formed for p2
    if p2_num_seq[1] <= 10:
        one_hot[25] = 1
    elif p2_num_seq[1] <= 20:
        one_hot[26] = 1
    else:
        one_hot[27] = 1

    # Two features for having J cards in hand
    for card in cards:
        if card["number"] == "J" and (card["suite"] == "spades" or card["suite"] == "hearts"):
            one_hot[28] = 1
        elif card["number"] == "J":
            one_hot[29] = 1

    return one_hot

class Square:
    """
    Implements the functionality of a square
    Each square should know:
        - what is the number and suite assigned to it
        - what is the status: color of the chip present in the square or empty
        - if a chip is present is it removable
    """
    def __init__(self, card):
        self.card = card
        self.status = "empty"
        self.isRemovable = True

        if self.card["number"] == "wild":
            self.status = "wild"
            self.isRemovable = False

    def place_chip(self, color):
        if self.status == "empty":
            self.status = color
        else:
            raise ValueError('Trying to place chip on non empty square!')

    def remove_chip(self):
        if self.isRemovable == True:
            self.status = "empty"
        else:
            raise ValueError('Trying to remove chip from a frozen square!')

    def __repr__(self):
        return "[{}-{}, {}]".format(self.card["number"][0], self.card["suite"][0], self.status[0])

class Board:
    """
    Captures the state of board
    """
    def __init__(self, size = 5, seq_len = 4, seq_count = 1):
        """
        Initializes a board of given size. Size 5 implies board of 5x5
        Currently works only for odd numbers (One wild in the center).
        To Do: Add support for even numbers (Corner wilds)
        """
        self.size = size
        self.seq_len = seq_len
        self.seq_count = seq_count
        self.board = [[None for i in range(self.size)] for j in range(self.size)]
        self.lookup = {}
        i = 0
        j = 0
        for _ in range(2):
            for n in available_numbers:
                for s in available_suites:
                    if i == int(size/2) and j == int(size/2):
                        card = {}
                        card["number"] = "wild"
                        card["suite"] = "wild"
                        self.board[i][j] = Square(card)
                        j += 1
                    card = {}
                    card["number"] = n
                    card["suite"] = s
                    self.board[i][j] = Square(card)
                    key = card["number"] + "_" + card["suite"]
                    if key in self.lookup.keys():
                        self.lookup[key].append((i, j))
                    else:
                        self.lookup[key] = []
                        self.lookup[key].append((i, j))
                    j += 1
                    if j == size:
                        i += 1
                        j = 0

    def check_sequence(self, color):
        """
        Check for existence of sequences and return True if it matches the number of required sequences
        """
        # Check rows
        count = 0
        for row in range(self.size):
            streak = 0
            for j in range(self.size):
                if self.get_status(row, j) == color:
                    streak += 1
                    if streak == self.seq_len:
                        streak = 0
                        count += 1
                        if count >= self.seq_count:
                            return True
        # Check cols
        for col in range(self.size):
            streak = 0
            for i in range(self.size):
                if self.get_status(i, col) == color:
                    streak += 1
                    if streak == self.seq_len:
                        streak = 0
                        count += 1
                        if count >= self.seq_count:
                            return True

        for s in range(2*self.size - 1):
            streak = 0
            for i in range(s + 1):
                j = s - i
                if i < self.size and j < self.size:
                    if self.get_status(i, j) == color:
                        streak += 1
                        if streak == self.seq_len:
                            streak = 0
                            count += 1
                            if count >= self.seq_count:
                                return True

        for d in range(-1*self.size + 1, self.size):
            streak = 0
            for j in range(0, 2*self.size - 1):
                i = j + d
                if i >= 0 and i < self.size and j < self.size:
                   if self.get_status(i, j) == color:
                       streak += 1
                       if streak == self.seq_len:
                           streak = 0
                           count += 1
                           if count >= self.seq_count:
                               return True

        return False

    def get_status(self, i, j):
        return self.board[i][j].status

    def is_removable(self, i, j):
        return self.board[i][j].isRemovable

    def place_chip(self, i, j, color):
        self.board[i][j].place_chip(color)

    def remove_chip(self, i, j):
        self.board[i][j].remove_chip()

    def get_empty_squares(self):
        """
        Return a list of coordinates empty squares on the board
        """
        empty_squares = []
        for i in range(self.size):
            for j in range(self.size):
                if self.get_status(i, j) == "empty":
                    empty_squares.append((i, j))
        return empty_squares

    def get_squares_by_chip_color(self, color):
        """
        Takes in an argument color and returns the list of coordinates of squares occupied by that color
        """
        color_squares = []
        for i in range(self.size):
            for j in range(self.size):
                if self.get_status(i, j) == color:
                    color_squares.append((i, j))
        return color_squares

    def __repr__(self):
        """ 
        Utility Function to print board
        To Do: Make this beautiful
        """
        board_str = ""
        for i in range(self.size):
            for j in range(self.size):
                board_str += str(self.board[i][j])
                board_str += " "
            board_str += "\n"
        return board_str

class Deck:
    """
    Captures the state of the deck
    """
    def __init__(self):
        self._deck = []
        for s in available_suites:
            for n in available_numbers:
                card = {}
                card["number"] = n
                card["suite"] = s
                self._deck.append(card)
                self._deck.append(card)
            card = {}
            card["number"] = "J"
            card["suite"] = s
            self._deck.append(card)
        random.shuffle(self._deck)

    def draw(self):
        return self._deck.pop()

class QLAgent:
    """
    reinforcement learning
    Class for q learning agent
    Implements train function to train the agent and move function that returns the recommended move given a state
    """
    def __init__(self, board_size, seq_len, num_seq, num_cards):
        """
        Initializes the required parameters of the game and initializes the Q Learning model
        """
        self.board_size = board_size
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.num_cards = num_cards
        self.num_actions = board_size*board_size*(len(available_numbers) + 1)*len(available_suites)
        qs = [0 for i in range(30)]
        b = [0 for i in range(self.num_actions)]
        W = [copy.deepcopy(qs) for i in range(self.num_actions)]
        self.model = {"W": W, "b": b}

    def move(self, board, deck, cards, moves):
        """
        Given the state of board, deck, cards in hand and a list of available moves returns the move with highest Q value
        """
        reco_move = moves[0]
        Q_curr = self.forward(get_features(board, cards), action_to_int(moves[0], self.board_size))
        for i in range(1, len(moves)):
            Q = self.forward(get_features(board, cards), action_to_int(moves[i], self.board_size))
            if Q > Q_curr:
                Q_curr = Q
                reco_move = moves[i]
        return reco_move

    def forward(self, features, a):
        """
        Forward pass on linear model
        """
        Q = dot(features, self.model["W"][a]) + self.model["b"][a] # Q = wx + b
        return Q

    def backward(self, features, a, delta, lr=0.1):
        """
        Backprop on model parameters
        """
        for i in range(len(features)):
            self.model["W"][a][i] += lr*delta*features[i] # Update only the weights of current action
        self.model["b"][a] += lr*delta

    def train(self, eps, gamma, lr=0.1):
        """
        Trains the Q learning agent. Takes hyperparametes - epsilon (degree of exploration), gamma (discount factor) and learning rate as arguments
        """
        board = Board(self.board_size, self.seq_len, self.num_seq)
        deck = Deck()
        p1 = Player("red")
        p2 = Player("blue")
        
        for _ in range(self.num_cards):
            p1.draw(deck)
            p2.draw(deck)
        result = p1.game_status
        while(result == "continue"):
            x = random.uniform(0, 1)
            moves = p1.available_moves(board)
            if len(moves) == 0:
                break
            if x < eps: # Epsilon for random action
                reco_move = random.choice(moves)
                Q_curr = self.forward(get_features(board, p1.show_cards()), action_to_int(reco_move, self.board_size))
            else: # Take greedy action
                reco_move = moves[0]
                Q_curr = self.forward(get_features(board, p1.show_cards()), action_to_int(moves[0], self.board_size))
                for i in range(1, len(moves)):
                    Q = self.forward(get_features(board, p1.show_cards()), action_to_int(moves[i], self.board_size))
                    if Q > Q_curr:
                        Q_curr = Q
                        reco_move = moves[i]
            board_ = copy.deepcopy(board)
            result = p1.play(reco_move, board_, deck)
            reward = 0 # Tiny reward For surviving so far
            if result != "continue":
                if result == "win":
                    reward = 1
                else:
                    reward = 0
            result = p2.random_play(board_, deck)
            if result != "continue":
                if result == "win":
                    Q_p_max = -1
                else:
                    Q_p_max = 0
            else:
                moves_ = p1.available_moves(board_)
                if len(moves_) == 0:
                    Q_p_max = 0
                else:
                    Q_p_max = self.forward(get_features(board_, p1.show_cards()), action_to_int(moves_[0], self.board_size))
                    for j in range(1, len(moves_)):
                        Q = self.forward(get_features(board_, p1.show_cards()), action_to_int(moves_[j], self.board_size))
                        if Q > Q_p_max:
                            Q_p_max = Q
            Q_new = reward + gamma*Q_p_max
            delta = Q_new - Q_curr
            self.backward(get_features(board, p1.show_cards()), action_to_int(reco_move, self.board_size), delta, lr=lr)
            board = copy.deepcopy(board_)
        return
        
        


class Player:
    """
    Models the player. 
    Keeps track of game status and cards in player's hand
    Can associate an agent to model the strategy of the player
    """
    def __init__(self, color, agent=None):
        self.color = color
        self._cards = []
        self.agent = agent
        self.game_status = "continue"

    def draw(self, deck):
        """
        Draws a card from deck and puts in player's hand
        """
        self._cards.append(deck.draw())

    def show_cards(self):
        # print(self._cards)
        return self._cards

    def available_moves(self, board):
        """
        Move can be characterized by card and location
        Given a board state returns the feasible moves for the player
        """
        moves = []
        for i in range(len(self._cards)):
            card = self._cards[i]
            if card["number"] != "J":
                key = card["number"] + "_" + card["suite"]
                locs = board.lookup[key]
                for loc in locs:
                    if board.get_status(loc[0], loc[1]) == "empty":
                        moves.append((self._cards[i], loc))
            else:
                if card["suite"] == "spades" or card["suite"] == "hearts": # Single eyed jacks
                    if self.color == "red":
                        opp_color = "blue"
                    else:
                        opp_color = "red"
                    locs = board.get_squares_by_chip_color(opp_color)
                    for loc in locs:
                        if board.is_removable(loc[0], loc[1]) == True:
                            moves.append((self._cards[i], loc))
                else: # Double eyed jacks
                    locs = board.get_empty_squares()
                    for loc in locs:
                        moves.append((self._cards[i], loc))
        return moves

    def play(self, move, board, deck):
        """
        Executes a move
        """
        card = move[0]
        loc = move[1]
        i = loc[0]
        j = loc[1]
        if card["number"] == "J" and (card["suite"] == "spades" or card["suite"] == "hearts"):
            board.remove_chip(i, j)
        else:
            board.place_chip(i, j, self.color)
        if board.check_sequence(self.color):
            self.game_status = "win"
            return self.game_status
        try:
            self.draw(deck)
        except:
            self.game_status = "draw"
            return self.game_status
        self.game_status = "continue"
        return self.game_status

    def random_play(self, board, deck):
        """
        Randomly select a move from the set of available moves
        """
        moves = self.available_moves(board)
        if len(moves) == 0:
            self.game_status = "draw"
            return self.game_status
        reco_move = random.choice(moves)
        return self.play(reco_move, board, deck)

    def strategic_play(self, board, deck):
        """
        Asks the agent to suggest the move and plays that
        If no agent is available returns random move
        """
        if self.agent:
            moves = self.available_moves(board)
            if len(moves) == 0:
                self.game_status = "draw"
                return self.game_status
            reco_move = self.agent.move(board, deck, self.show_cards(), moves)
            return self.play(reco_move, board, deck)
        else:
            print("Agent not available returning a random move")
            return self.random_play(board, deck)
    
    def associate_agent(self, agent):
        """
        Associate an agent to the player
        """
        if self.agent != None:
            print("Warning: Overwriting an agent")
        self.agent = agent


def simulate_random(size, seq_len, num_seq, num_cards):
    """
    Simulate a random game with the given set of parameters
    """
    board = Board(size, seq_len, num_seq)
    deck = Deck()
    p1 = Player("red")
    p2 = Player("blue")
    for _ in range(num_cards):
        p1.draw(deck)
        p2.draw(deck)
    turn = 0
    while True:
        # print(board)
        if turn == 0:
            result = p1.random_play(board, deck)
            if result == "win":
                return 1
            elif result == "draw":
                return 0
            turn = 1
        else:
            result = p2.random_play(board, deck)
            if result == "win":
                return -1
            elif result == "draw":
                return 0
            turn = 0

def simulate_strategic(size, seq_len, num_seq, num_cards, agent):
    """
    Simulates a game with player 1 playing according to the agent and player 2 playing randomly
    """
    board = Board(size, seq_len, num_seq)
    deck = Deck()
    p1 = Player("red", agent) # Create P1 with learned agent
    p2 = Player("blue")
    for _ in range(num_cards):
        p1.draw(deck)
        p2.draw(deck)
    turn = 0
    while True:
        # print(board)
        if turn == 0:
            result = p1.strategic_play(board, deck)
            if result == "win":
                return 1
            elif result == "draw":
                return 0
            turn = 1
        else:
            result = p2.random_play(board, deck)
            if result == "win":
                return -1
            elif result == "draw":
                return 0
            turn = 0

def test():
    """
    Some unit tests
    """
    board = Board(5, 4, 1)
    deck = Deck()
    # print(len(deck._deck))
    # for _ in range(3):
        # print(deck.draw())
    # print(len(deck._deck))
    p1 = Player("red")
    p2 = Player("blue")
    p1.draw(deck)
    p2.draw(deck)
    p1.draw(deck)
    p2.draw(deck)
    p1.show_cards()
    p2.show_cards()
    # print(len(deck._deck))
    for _ in range(4):
        moves = p1.available_moves(board)
        # print(moves[0])
        p1.play(moves[0], board, deck)
    # print(board)
    # print(board.check_sequence("red"))
    get_features(board, p1.show_cards())

def main():
    # test()
    # sys.exit()
    # random.seed(5)
    num_games = 10000
    board_size = 5
    seq_len = 4
    num_seq = 1
    num_cards = 3
    tlimit = 120 # Time limit to train Q Learning agent (in seconds)

    ### Test Random Agent ###
    win = 0
    loss = 0
    draw = 0
    for _ in range(num_games):
        r = simulate_random(board_size, seq_len, num_seq, num_cards)
        if r == 1:
            win += 1
        elif r == 0:
            draw += 1
        else:
            loss += 1
    print("Random Play {} games: win/loss/draw: {}/{}/{}".format(num_games, win, loss, draw))

    ### Train a Q Learning agent ###
    
    start = time.time()
    eps = 0.7
    gamma = 0.9
    lr = 0.1
    agent = QLAgent(board_size, seq_len, num_seq, num_cards)
    
    count = 0
    while(time.time() < start + tlimit - 0.1):
    # for i in range(50):
        agent.train(eps, gamma, lr)
        count += 1
        if count%2000 == 0:
            lr = lr/2

    print("Trained agent for {} Iterations".format(count))

    ### Test Q Learning agent ###
    win = 0
    loss = 0
    draw = 0
    for _ in range(num_games):
        r = simulate_strategic(board_size, seq_len, num_seq, num_cards, agent)
        if r == 1:
            win += 1
        elif r == 0:
            draw += 1
        else:
            loss += 1
    print("Strategic Play {} games: win/loss/draw: {}/{}/{}".format(num_games, win, loss, draw))


if __name__ == "__main__":
    main()
