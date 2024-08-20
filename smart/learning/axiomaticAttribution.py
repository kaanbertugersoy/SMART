import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance


from scipy import stats
import pandas as pd


# For the purpose of FEATURE IMPORTANCE analysis, we use the Integrated Gradients
# which is an axiomatic attribution method that assigns an importance score to each
# input feature by approximating the integral of the gradients of the model's output
# with respect to the input features along the path from a baseline input to the
# input of interest. The Integrated Gradients method is based on the axioms of
# Sensitivity, Implementation Invariance, Completeness, and Linearity.


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


def define_model(input_dim):
    torch.manual_seed(42)

    class NNModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, 12)
            self.sigmoid1 = nn.Sigmoid()
            self.linear2 = nn.Linear(12, 8)
            self.sigmoid2 = nn.Sigmoid()
            self.linear3 = nn.Linear(8, 2)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            lin1_out = self.linear1(x)
            sigmoid1_out = self.sigmoid1(lin1_out)
            lin2_out = self.linear2(sigmoid1_out)
            sigmoid2_out = self.sigmoid2(lin2_out)
            lin3_out = self.linear3(sigmoid2_out)
            softmax_out = self.softmax(lin3_out)
            return softmax_out

    return NNModel(input_dim)


def train_model(model, train_loader, x):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total_loss, total_acc = list(), list()
    feat_imp = np.zeros(x.shape[1])
    num_epochs = 100

    for epoch in range(num_epochs):
        losses = 0
        for idx, (x, y) in enumerate(train_loader):
            x, y = x.float(), y.type(torch.LongTensor)
            x.requires_grad = True
            optimizer.zero_grad()
            preds = model.forward(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            losses += loss.item()
        total_loss.append(losses/len(train_loader))
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}: TLoss: {total_loss[-1]}")
    return model

# torch.save(model.state_dict(
# ), "C:/Users/LENOVO/Desktop/SMART/smart/checkpoints/stock_market_model.pth")


def ig_attribute(model, x):
    test_input_tensor = torch.from_numpy(x.values).type(torch.FloatTensor)

    ig = IntegratedGradients(model)

    test_input_tensor.requires_grad_()
    attr, delta = ig.attribute(test_input_tensor, target=1,
                               return_convergence_delta=True)
    return attr


def comprehend(x, cols):
    model = define_model(input_dim=len(cols))

    feat_imp = np.mean(
        np.abs(ig_attribute(model, x).detach().numpy()), axis=0)

    importance_values = [(a, b) for a, b in sorted(zip(feat_imp, cols))]

    return importance_values, feat_imp, model


# Helper functions

def test_results(model, test_loader):
    model.eval()
    correct = 0

    for idx, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            x, y = x.float(), y.type(torch.LongTensor)
            pred = model(x)
            preds_class = torch.argmax(pred)
            if (preds_class.numpy() == y.numpy()[0]):
                correct += 1
    print("Accuracy: ", correct/len(test_loader))
    return correct/len(test_loader)


def visualize_importances(feature_names, importances, title="Stock Market Model Feature Importances", plot=True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", importances[i])
    x_pos = np.arange(len(feature_names))
    if plot:
        plt.figure(figsize=(12, 6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)
        plt.show()
