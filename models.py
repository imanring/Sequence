from torch import nn

class cnn_scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(5,5), stride=1, padding='same')
        self.act1 = nn.ReLU()
        #self.drop1 = nn.Dropout(0.3)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3,3), stride=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()
        self.fc3 = nn.Linear(16, 25)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.15)
        self.fc4 = nn.Linear(25, 1)
        
    def forward(self,board):
        """
        Score the state to pick a move.
        Parameters
        ----------
        board: [3x10x10]
            State of the board with tokens. 'black' tokens are in the first layer, 'white' tokens are in the second layer
        """
        b = self.conv1(board)
        b = self.pool1(self.act1(b))
        b = self.conv2(b)
        b = self.pool2(self.act2(b))
        b = self.flat(b)
        o = self.fc3(b)
        o = self.act3(o)
        o = self.drop3(o)
        o = self.fc4(o)
        return o


class dnn_scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Linear(14, 1,bias=False)
    
    def forward(self,x):
        return self.d1(x)