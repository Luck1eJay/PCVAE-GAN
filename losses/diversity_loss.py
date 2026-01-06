import torch

def diversity_loss(phi_list):
    """
    Encourage diversity among multiple solutions
    """
    N = len(phi_list)
    if N <= 1:
        return torch.tensor(0.0, device=phi_list[0].device)

    loss = 0.0
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            loss += torch.mean(torch.abs(phi_list[i] - phi_list[j]))
            count += 1

    return - loss / count
