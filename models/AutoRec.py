import torch

class AutoRec(torch.nn.Module):
    def __init__(self, input_size, emb_size, dropout=0.1, bias=True):
        super().__init__()
        self.inp_layer = torch.nn.Linear(input_size, emb_size, bias=bias)
        self.out_layer = torch.nn.Linear(emb_size, input_size, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.sigmoid = torch.nn.Sigmoid()

    def encode(self, x):
        return self.sigmoid(self.inp_layer(x))

    def decode(self, emb):
        return self.out_layer(emb)

    def forward(self, x):
        emb = self.encode(x)
        emb = self.dropout(emb)
        return self.decode(emb)
