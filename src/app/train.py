import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error


class Train:

    def __init__(self, preobj, *, learning_rate=1e-4, epoch=10):
        self.preobj = preobj
        self.device = torch.device('cpu')
        self.dtype = torch.float
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.X_train, self.X_val, self. y_train, self.y_val = self.make_tensors(self.preobj, self.device, self.dtype)
        self.model = self.init_model(self.X_train.shape[1], 10)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.loss_hist_train, self.loss_hist_val, self.trained_model = self.train_model(self.model, self.epoch, self.optimizer,
                                                                    self.criterion, self.X_train, self.y_train,
                                                                    self.X_val, self.y_val)
    @classmethod
    def make_tensors(self, preobj, device, dtype):
        x_train = torch.as_tensor(preobj.splits['X_train'], device=device, dtype=dtype)
        x_val = torch.as_tensor(preobj.splits['X_val'], device=device, dtype=dtype)
        y_train = torch.as_tensor(preobj.splits['y_train'], device=device, dtype=torch.long)
        y_val = torch.as_tensor(preobj.splits['y_val'], device=device, dtype=torch.long)
        return x_train, x_val, y_train, y_val

    def init_model(self, D_in, D_out):
        # Logistic regression model
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, D_out)
        )
        return model

    def train_model(self, model, epoch, optimizer, criterion, x_train, y_train, x_val, y_val):
        loss_hist = []
        loss_hist_val = []
        # Train
        for t in range(epoch):
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs_val = model(x_val)
            loss_val = criterion(outputs_val, y_val)

            if t % 10 == 0:
                loss_hist.append(loss.item())
                loss_hist_val.append(loss_val.item())
                print(t, loss.item(), loss_val.item())
        torch.save(model, '../../model/softmax')

        return loss_hist, loss_hist_val, model

