import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is None:
        return None
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def zero_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is None:
        return None
    if isinstance(h, torch.Tensor):
        return h * 0
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target

def cal_acc(labels, targets):
    # print("label size:", labels.size(), "target size:", targets.size())
    # Calculate the argmax along the appropriate dimension
    output_labels = torch.argmax(labels, dim=1)
    # target_labels = torch.argmax(targets, dim=1)
    target_labels = targets

    # Calculate the number of correct predictions
    correct = (output_labels == target_labels).sum().item()

    # Calculate the total number of predictions
    total = labels.size(0)

    # Calculate the accuracy
    accuracy = correct / total
    
    return accuracy

def cal_tag_acc(labels, targets):
    # Calculate the argmax along the appropriate dimension
    output_labels = torch.argmax(labels, dim=1)
    # target_labels = torch.argmax(targets, dim=1)
    target_labels = targets

    # Ignore samples where target label is the last index
    valid_indices = target_labels != (labels.size(1) - 1)
    output_labels = output_labels[valid_indices]
    target_labels = target_labels[valid_indices]

    # Calculate the number of correct predictions
    correct = (output_labels == target_labels).sum().item()

    # Calculate the total number of predictions
    total = len(valid_indices.nonzero())

    # Calculate the accuracy
    accuracy = correct / total if total > 0 else 1.0
    
    return accuracy

