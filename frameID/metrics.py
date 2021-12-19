import torch
import torch.nn as nn
import torch.nn.functional as F

BIG_NUMBER = 1e9

class ContrastiveLoss(nn.Module):

    def __init__(self, h_norm=True, temperature=1.0):

        super(ContrastiveLoss, self).__init__()
        self.h_norm = h_norm
        self.temperature = temperature

    def forward(self, x):

        if self.h_norm:
            x = F.normalize(x, dim=-1)

        h1, h2 = torch.split(x, x.shape[0] // 2, dim=0)
        batch_size = h1.shape[0]

        labels = torch.arange(batch_size)
        masks = F.one_hot(torch.arange(batch_size), batch_size)

        # Calculate cosines between different images in first set.
        logits_aa = torch.matmul(h1, h1.T) / self.temperature
        logits_aa = logits_aa - masks * BIG_NUMBER

        # Calculate cosines between different images in second set.
        logits_bb = torch.matmul(h2, h2.T) / self.temperature
        logits_bb = logits_bb - masks * BIG_NUMBER

        # Calculate cosines between images in different sets.
        logits_ab = torch.matmul(h1, h2.T) / self.temperature
        logits_ba = logits_ab.T

        loss_a = F.cross_entropy(torch.concat((logits_ab, logits_aa), dim=-1), labels)
        loss_b = F.cross_entropy(torch.concat((logits_ba, logits_bb), dim=-1), labels)

        loss = torch.mean(loss_a + loss_b)

        return loss, logits_ab, labels

if __name__ == '__main__':

    h = torch.randn([32, 16])

    criterion = ContrastiveLoss()

    loss, logits, labels = criterion(h)

    print(loss)
    print(logits.shape)
    print(labels.shape)
