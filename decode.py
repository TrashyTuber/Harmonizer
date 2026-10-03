import torch

def viterbi_decode(logits, switch_penalty):
    logits = logits.cpu()
    T, C = logits.shape

    dp = logits[0].clone()

    backpointer = torch.empty((T, C), dtype=torch.long)
    stay_idx = torch.arange(C)

    for t in range(1, T):
        m, m_idx = dp.max(0)
        switch_score = m - switch_penalty
        stay_wins = dp >= switch_score
        backpointer[t] = torch.where(stay_wins, stay_idx, m_idx)
        dp = logits[t] + torch.where(stay_wins, dp, switch_score)

    path = [int(dp.argmax())]

    for t in range(T - 1, 0, -1):
        path.append(int(backpointer[t, path[-1]]))
        
    return path[::-1]

def transition_viterbi_decode(logits, logT, lam):
    logits = logits.cpu()
    T, C = logits.shape

    dp = logits[0].clone()

    backpointer = torch.empty((T, C), dtype=torch.long)

    for t in range(1, T):
        scores = dp[:, None] + lam * logT
        vals, idx = scores.max(dim=0)
        backpointer[t] = idx
        dp = logits[t] + vals

    path = [int(dp.argmax())]

    for t in range(T - 1, 0, -1):
        path.append(int(backpointer[t, path[-1]]))
        
    return path[::-1]

    
    
