import torch
import torch.nn as nn

# fix seed so that random initialization always performs the same 
torch.manual_seed(1)


# create the model N as described in the 
N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))

x = torch.rand((1, 10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data
x.requires_grad_() # make sure we can compute the gradient w.r.t x
t = 1 # target class
eps_real = 0.4 #depending on your data this might be large or small

eps = eps_real - 1e-7 # small constant to offset floating-point erros


def print_class(str, input_x, y_expected):
    y_predicted = N(input_x)
    print("{} class: {}".format(str, y_predicted.argmax(dim=1).item()))
    assert (y_predicted.argmax(dim=1).item() == y_expected)

print_class("Original", x, 2)

# compute gradient
# note that CrossEntropyLoss() combines the cross-entropy loss and an implicit softmax function
L = nn.CrossEntropyLoss()
loss = L(N(x), torch.tensor([t], dtype=torch.long))
loss.backward()

# your code here
# in x.grad you have access to the gradient of loss w.r.t. x
eta = eps * torch.sign(x.grad) # perturbation eta
x_bar = x - eta # the perturbed example

print_class("New", x_bar, 1)
assert(torch.norm((x - x_bar), p=float('inf')) <= eps_real)
