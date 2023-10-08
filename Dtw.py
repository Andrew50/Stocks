#from soft_dtw_cuda.soft_dtw_cuda import SoftDTW
#import torch

## Create the sequences
#batch_size, len_x, len_y, dims = 8, 15, 12, 5
#x = torch.rand((batch_size, len_x, dims), requires_grad=True)
#y = torch.rand((batch_size, len_y, dims))
## Transfer tensors to the GPU
#x = x.cuda()
#y = y.cuda()

## Create the "criterion" object
##alignment_score = soft_dtw.fit(X, Y)
#sdtw = SoftDTW(use_cuda=True, gamma=0.1)

## Compute the loss value
#loss = sdtw(x, y)  # Just like any torch.nn.xyzLoss()

## Aggregate and call backward()
from struct import pack
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Soft DTW function with CUDA support
class SoftDTW(nn.Module):
    def __init__(self, gamma=1.0):
        super(SoftDTW, self).__init__()
        self.gamma = gamma

    def forward(self, D):
        N, M = D.shape
        D = torch.exp(-D / self.gamma).cuda()  # Move data to GPU

        acc_cost = torch.zeros((N, M)).cuda()  # Initialize on GPU
        acc_grad = torch.zeros((N, M)).cuda()  # Initialize on GPU

        for i in range(N):
            for j in range(M):
                if i > 0 and j > 0:
                    min_val, _ = torch.min(torch.stack([acc_cost[i - 1, j], acc_cost[i, j - 1], acc_cost[i - 1, j - 1]]), dim=0)
                    acc_cost[i, j] = min_val + D[i, j]
                elif i > 0:
                    acc_cost[i, j] = acc_cost[i - 1, j] + D[i, j]
                elif j > 0:
                    acc_cost[i, j] = acc_cost[i, j - 1] + D[i, j]
                else:
                    acc_cost[i, j] = D[i, j]

                d = torch.exp(-D[i, j] / self.gamma)
                if i > 0 and j > 0:
                    acc_grad[i - 1, j - 1] += (acc_grad[i, j] + d) / self.gamma
                if i > 0:
                    acc_grad[i - 1, j] += (acc_grad[i, j] + d) / self.gamma
                if j > 0:
                    acc_grad[i, j - 1] += (acc_grad[i, j] + d) / self.gamma
                acc_grad[i, j] -= (acc_grad[i, j] + d) / self.gamma

        return acc_cost[N - 1, M - 1], acc_grad


def dtw(x,y):
    print('-1')
    sequence1 = torch.tensor(x, requires_grad=True).cuda()
    sequence2 = torch.tensor(y, requires_grad=True).cuda()
    print('0')
    # Compute the pairwise distance matrix on the GPU
    print('1')
    pairwise_distance = torch.abs(sequence1.unsqueeze(1) - sequence2.unsqueeze(0)).cuda()

    # Create the Soft DTW model on the GPU
    soft_dtw = SoftDTW(gamma=1.0)
    print('2')
    # Calculate the Soft DTW distance and gradients on the GPU
    cost, grad = soft_dtw(pairwise_distance)
    print('3')

    # Compute gradients with respect to input sequences
   # sequence1_grad = torch.autograd.grad(cost, sequence1, retain_graph=True)[0]
    #sequence2_grad = torch.autograd.grad(cost, sequence2, retain_graph=True)[0]

    return cost.item()

# Example usage
if __name__ == '__main__':
    ## Create two sequences on the GPU
    #sequence1 = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True).cuda()
    #sequence2 = torch.tensor([1.0, 2.0, 2.5, 3.5], requires_grad=True).cuda()
    ## Compute the pairwise distance matrix on the GPU
    #pairwise_distance = torch.abs(sequence1.unsqueeze(1) - sequence2.unsqueeze(0)).cuda()

    ## Create the Soft DTW model on the GPU
    #soft_dtw = SoftDTW(gamma=1.0)

    ## Calculate the Soft DTW distance and gradients on the GPU
    #cost, grad = soft_dtw(pairwise_distance)

    ## Compute gradients with respect to input sequences
    #sequence1_grad = torch.autograd.grad(cost, sequence1, retain_graph=True)[0]
    #sequence2_grad = torch.autograd.grad(cost, sequence2, retain_graph=True)[0]

   # print("Soft DTW Cost:", cost.item())
    x = [1.0, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 3.0, 4.0]
    #rint("Soft DTW Cost:",dtw(x,y))
    #rint("Gradient of Sequence 1:", sequence1_grad)
    #rint("Gradient of Sequence 2:", sequence2_grad)
    
