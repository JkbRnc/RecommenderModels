import torch
from torch.utils.data import DataLoader


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
        self.out_layer(emb)

    def forward(self, x):
        emb = self.encode(x)
        emb = self.dropout(emb)
        return self.decode(emb)

    def fit(
        self,
        train_data,
        criterion,
        optim,
        max_epochs=300,
        batch_size=64,
        valid_data=None,
    ):
        self.optim = optim
        self.criterion = criterion
        validate = False
        if valid_data is not None:
            validate = True
            valid_loss = []
            valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

        train_loss = []
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        for epoch in range(max_epochs):
            print("=" * 10, "Epoch:", epoch + 1, "=" * 10)
            loss = self._train_epoch(train_loader)
            train_loss.extend(loss)
            print("Train loss:", sum(loss) / len(loss))
            if validate:
                loss = self._valid_epoch(valid_loader)
                valid_loss.extend(loss)
                print("Valid loss:", sum(loss) / len(loss))

    def _train_epoch(self, data):
        list_loss = []
        self.train()
        for item in data:
            r = item.float()
            r_hat = self.forward(r)
            loss = self.criterion(r, r_hat * torch.sign(r))

            list_loss.append(loss.item())
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return list_loss

    def _valid_epoch(self, data):
        list_loss = []
        self.eval()
        with torch.no_grad():
            for item in data:
                r = item.float()
                r_hat = self.forward(r)
                loss = self.criterion(r, r_hat * torch.sign(r))
                list_loss.append(loss.item())

        return list_loss
