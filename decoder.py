import torch
import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, config, args):

        super(Classifier, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_labels = args.num_labels
        self.current_device = args.device
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, claim_emb_list, evid_emb, labels):

        logits = []
        for label_id in range(len(claim_emb_list)):
            logits_one_label = torch.sum(torch.multiply(claim_emb_list[label_id], evid_emb), dim=-1)
            logits.append(torch.reshape(logits_one_label, [-1, 1]))
        logits = torch.concat(logits, dim=-1)
        y_pred = torch.argmax(logits, dim=-1)
        one_hot = nn.functional.one_hot(labels.to(self.current_device), num_classes=self.num_labels)
        one_hot = one_hot.float()
        y_pred_prob = self.softmax(logits)
        y_pred_prob = torch.clamp(y_pred_prob, min=1e-12)
        loss = - torch.sum(torch.multiply(one_hot, torch.log(y_pred_prob)), dim=-1)
        # loss = self.ce_loss(logits, one_hot, reduction='none')
        loss = torch.mean(loss)

        return loss, y_pred