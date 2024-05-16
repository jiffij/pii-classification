import argparse
import functools
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import pytorch_warmup as warmup
from transformers import MobileBertForTokenClassification, MobileBertConfig

import data
import model

from utils import batchify, get_batch, repackage_hidden, zero_hidden, cal_acc, cal_tag_acc

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU, BERT)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=1024,
                    help='sequence length')
parser.add_argument('--warmup', type=int, default=1000,
                    help='warmup for learning rate')
parser.add_argument('--cooldown', type=int, default=None,
                    help='cooldown for learning rate')
parser.add_argument('--accumulate', type=int, default=1,
                    help='number of batches to accumulate before gradient update')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.0,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--ce_weight', action='store_true',
                    help='use Cross Entropy Weight')
parser.add_argument('--language_model', action='store_true',
                    help='pretrain with language modeling')
parser.add_argument('--criterion', type=str, default='MSE',
                    help='MSE or CosineSimilarity')
args = parser.parse_args()
args.tied = True

if __name__ == '__main__': 
    # Set the random seed manually for reproducibility.
    if args.cuda:
        print("."*10, "using cuda", "."*10)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    def model_save(fn):
        with open(fn, 'wb') as f:
            #torch.save([model, criterion, optimizer], f)
            torch.save([model, criterion, optimizer, lr_scheduler, warmup_scheduler], f)

    def model_load(fn):
        global model#, criterion, optimizer
        with open(fn, 'rb') as f:
            #torch.nn.Module.dump_patches = True
            #model, criterion, optimizer = torch.load(f)
            #model, criterion = torch.load(f)
            # m, criterion = torch.load(f)
            m, criterion, optimizer, lr_scheduler, warmup_scheduler = torch.load(f)
            d = m.state_dict()
            #del d['pos_emb']
            model.load_state_dict(d, strict=False)
            print("pretrained model loaded...")
            if False:
                for block in model.blocks:
                    print(block.attn)
                    if block.attn: block.attn.vq_collapse()
            del m

    import os
    import hashlib
    from pii import PII, PII_MODES
    # fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    # if os.path.exists(fn):
    #     print('Loading cached dataset...')
    #     corpus = torch.load(fn)
    # else:
    #     print('Producing dataset...')
    #     corpus = data.Corpus(args.data)
    #     torch.save(corpus, fn)

    # eval_batch_size = min(100, args.batch_size)
    # print('Eval batch size of', eval_batch_size)
    # test_batch_size = 8
    # train_data = batchify(corpus.train, args.batch_size, args)
    # val_data = batchify(corpus.valid, eval_batch_size, args)
    # test_data = batchify(corpus.test, test_batch_size, args)
    
    pii = PII(os.path.join('..', 'dataset'), use_cuda=args.cuda, language_model=args.language_model,
              use_fast=False if args.model == "BERT" else True)
    

    ###############################################################################
    # Build the model
    ###############################################################################

    from splitcross import SplitCrossEntropyLoss
    criterion = None

    ntokens = pii.vocab_dim
    print('Total number of tokens:', ntokens)
    if args.model == "BERT":
        # cfg = MobileBertConfig(vocab_size=ntokens, hidden_size=1024, intermediate_size=1024, max_position_embeddings=1024,
        #                        embedding_size=1024, intra_bottleneck_size=1024, num_labels=13)
        cfg = MobileBertConfig(vocab_size=ntokens, num_labels=13, max_position_embeddings=1024)
        model = MobileBertForTokenClassification(cfg)
        print(model)
    #model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
    #model = model.BoomRNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
    else:
        model = model.SHARNN(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied, language_model=args.language_model)
    #model = model.AttnRNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
    #model = model.RecAttn(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
    #model = model.LNRNN(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
    #model = model.LNRR(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
    ###
    if args.resume and args.epochs > 0:
        print(f'Resuming model {args.resume} ...')
        model_load(args.resume)
        #optimizer.param_groups[0]['lr'] = args.lr
        model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
        #if args.wdrop:
        #    from weight_drop import WeightDrop
        #    for rnn in model.rnns:
        #        if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
        #        elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
    ###
    if not criterion:
        splits = []
        if ntokens > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif ntokens > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
        print('Using', splits)
        if args.language_model:
            # criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
            if args.criterion == "CosineSimilarity":
                criterion = nn.CosineSimilarity(dim=1, eps=1e-8)
            else:
                criterion = nn.MSELoss()
        # TODO try to change to cross-entropy
        else:
            criterion = nn.CrossEntropyLoss(weight=pii.class_weight if args.ce_weight else None, reduction='mean')
    ###
    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        
    if False: # or args.jit:
        print('Jitting ...')
        model.eval()
        model.lmr = torch.jit.trace(model.lmr, (torch.rand([args.bptt, args.batch_size, args.emsize]).cuda(), torch.rand([1, args.batch_size, args.emsize]).cuda()))
    #model = torch.jit.trace_module(model, torch.zeros((args.bptt, args.batch_size), dtype=torch.long))
    ###
    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Args:', args)
    print('Model total parameters:', total_params)

    ###############################################################################
    # Training code
    ###############################################################################

    def evaluate():
        # Turn on evaluation mode which disables dropout.
        pii.mode = PII_MODES[1] # added
        model.eval()
        if args.model == 'QRNN' and getattr(model, 'reset', None): model.reset()
        total_loss = 0
        total_acc = 0
        total_tag_acc = 0
        # ntokens = len(pii.dictionary)
        hidden = None
        mems = None
        notag_batches = 0
        with torch.no_grad():
            for i in range(len(pii)): #range(0, data_source.size(0) - 1, args.bptt):
                # data, targets = get_batch(data_source, i, args, evaluation=True)
                data, targets = pii[i]
                #output, hidden = model(data, hidden)
                if args.model == "BERT":
                    logits = model(input_ids=data.unsqueeze(0)).logits.squeeze(0)
                else:
                    output, hidden, mems, logits = model(data, hidden, mems=mems, return_h=False)
                # TODO changed
                if args.language_model:
                    # total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets.view(-1)).data
                    if args.criterion == "CosineSimilarity":
                        output = F.normalize(output, dim=1)
                        targets = F.normalize(targets, dim=1)
                        total_loss += (1 - criterion(output, targets)).sum()
                    else:
                        total_loss += criterion(output, targets)
                else:
                    total_loss += criterion(logits.float(), targets)
                    total_acc += cal_acc(logits, targets)
                    ta = cal_tag_acc(logits, targets)
                    total_tag_acc += 0 if ta == -1 else ta
                    notag_batches += 1 if ta == -1 else 0
                if hidden is not None:
                    hidden = repackage_hidden(hidden)
        return total_loss.item() / len(pii), total_acc/len(pii), total_tag_acc/(len(pii)-notag_batches)


    ## I added
    steps_per_epoch = len(pii) // args.accumulate
    warmup_period = args.warmup
    num_steps = steps_per_epoch * args.epochs - warmup_period
    t0 = num_steps // 3
    lr_min = 0
    max_step = t0 * 3 + warmup_period


    def train(epoch=0):
        # Turn on training mode which enables dropout.
        pii.mode = PII_MODES[0] # added
        if args.model == 'QRNN' and getattr(model, 'reset', None): model.reset()
        total_loss = 0
        total_acc = 0
        total_tag_acc = 0
        start_time = time.time()
        # ntokens = len(pii.dictionary)
        hidden = None
        mems = None
        batch, i = 0, 0
        loss_every_n_batches = args.accumulate
        losses = []
        # changed
        pii.mode = PII_MODES[0]
        notag_batches = 0
        for i in range(len(pii)):
            # TODO the bptt is incorrect, we are using dynamic input size
            # Warmup
            
            
            # for param_group in optimizer.param_groups:
            #     step = epoch * len(pii) + batch + 1 #(len(pii) // args.bptt) + batch + 1 # changed
            #     pctwarm = min(step, args.warmup) / args.warmup
            #     if args.cooldown:
            #         pctcool = max(min(step - args.cooldown, args.cooldown) / args.cooldown, 0)
            #     else:
            #         pctcool = 0
            #     param_group['lr'] = args.lr * (pctwarm - pctcool)
                
                
                #param_group['betas'] = (0.95 - (pctwarm - pctcool) * 0.05, param_group['betas'][1])
            if True:
                bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
                # Prevent excessively small or negative sequence lengths
                seq_len = max(5, int(np.random.normal(bptt, 5)))
                # There's a very small chance that it could select a very long sequence length resulting in OOM
                seq_len = min(seq_len, args.bptt)
            else:
                seq_len = args.bptt
            #print(seq_len)

            #lr2 = optimizer.param_groups[0]['lr']
            #optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
            model.train()
            
            # changed
            # data, targets = get_batch(train_data, i, args, seq_len=seq_len)
            data, targets = pii[i]

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            #optimizer.zero_grad()

            if args.wdrop and args.model != "BERT":
                rnn_hh_weights = []
                rnn_hh_masks = []
                for b in model.blocks:
                    # Create a mask with wdrop entries as zeros
                    m = ((1 - args.wdrop) * torch.ones(b.rnn.weight_hh_l0.shape, device=b.rnn.weight_hh_l0.device)).bernoulli()
                    rnn_hh_masks.append(m)
                    # Knock out all of those weights
                    wd = m * b.rnn.weight_hh_l0.data
                    # Save the original weights
                    rnn_hh_weights.append(b.rnn.weight_hh_l0.data)
                    # Replace the weight to be used and scale it up
                    b.rnn.weight_hh_l0.data = wd / (1 - args.wdrop)
                    b.rnn.flatten_parameters()

            #output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
            #output, hidden, mems, attn_outs, _ = model(data, hidden, return_h=True, mems=mems)
            if args.language_model:
                output, hidden, mems, logits, attn_outs, _ = model(data, hidden, return_h=True, mems=mems)
                # raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets.view(-1))
                if args.criterion == "CosineSimilarity":
                    output = F.normalize(output, dim=1)
                    targets = F.normalize(targets, dim=1)
                    raw_loss = (1 - criterion(output, targets)).sum()
                else:
                    raw_loss = criterion(output, targets)
            elif args.model == "BERT":
                logits = model(input_ids=data.unsqueeze(0)).logits.squeeze(0)
                raw_loss = criterion(logits.float(), targets)
            else:
                output, hidden, mems, logits, attn_outs, _ = model(data, hidden, return_h=True, mems=mems)
                raw_loss = criterion(logits.float(), targets) # criterion(model.decoder.weight, model.decoder.bias, output, targets.view(-1))

            if not args.language_model:
                # ACC
                acc = cal_acc(logits, targets)
                # total_tag_acc += cal_tag_acc(logits, targets)
                ta = cal_tag_acc(logits, targets)
                total_tag_acc += 0 if ta == -1 else ta
                notag_batches += 1 if ta == -1 else 0
                # print("train_acc:", cal_acc(logits, targets), "raw loss:", raw_loss)
            
            losses.append(raw_loss)

            if False and mems:
                mem_loss = sum(args.alpha * m.pow(2).mean() for m in mems)
                losses.append(mem_loss)

            #print(output.shape, targets.shape)
            #next_targets = targets.view(len(output), -1)[1:].view(-1)
            #print(output[:-1].shape, next_targets.shape)
            #next_token_loss = 0.1 * criterion(model.decoder.weight, model.decoder.bias, output[:-1], next_targets)
            # Activiation Regularization
            #if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            #if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            '''
            if attn_outs:
                outs = []
                attn_out = torch.stack(attn_outs)
                for a in range(len(attn_out)):
                    for b in range(len(attn_out)):
                        if a == b: continue
                    outs.append(F.cosine_similarity(attn_out[a], attn_out[b], dim=-1).mean())
                attn_loss = functools.reduce(lambda x, y: x + y, outs) / len(outs)
                # We want the vectors to be dissimilar - if they're one they're similar - so let's flip it
                losses.append(-attn_loss)
            '''

            if batch % loss_every_n_batches == 0:
                loss = functools.reduce(lambda x, y: x + y, losses)
                #print(losses)
                # loss.backward()
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                if args.criterion == "CosineSimilarity":
                    loss.sum().backward()
                else:
                    loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
                optimizer.step()
                #TODO I added these
                with warmup_scheduler.dampening():
                    if warmup_scheduler.last_step + 1 >= warmup_period:
                        lr_scheduler.step()
                if warmup_scheduler.last_step + 1 >= max_step:
                    break
                
                if hidden is not None:
                    #if np.random.random() > 0.975:
                    #    hidden = None
                    #    #hidden = zero_hidden(hidden)
                    hidden = repackage_hidden(hidden)
                if mems is not None:
                    #if np.random.random() > 0.975:
                    #    mems = None
                    #    mems = zero_hidden(mems)
                    mems = repackage_hidden(mems)
                optimizer.zero_grad()
                losses = []

            if args.wdrop and args.model != "BERT":
                for (w, m, b) in zip(rnn_hh_weights, rnn_hh_masks, model.blocks):
                    # Scale the resulting weight back down
                    b.rnn.weight_hh_l0.data = b.rnn.weight_hh_l0.data * (1 - args.wdrop)
                    # Replace the zeroed entries with their original values
                    m = m.type(torch.bool)
                    b.rnn.weight_hh_l0.data[~m] = w[~m]
                    b.rnn.flatten_parameters()

            if args.language_model:
                total_loss += raw_loss
            else:
                total_loss += raw_loss #raw_loss.data #TODO changed
                total_acc += acc

            #optimizer.param_groups[0]['lr'] = lr2
            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss.item() / args.log_interval
                cur_acc = total_acc / args.log_interval
                cur_tag_acc = total_tag_acc / (args.log_interval-notag_batches)
                elapsed = time.time() - start_time
                try:
                    ex = math.exp(cur_loss)
                except OverflowError:
                    ex = cur_loss
                try:
                    cur = cur_loss / math.log(2)
                except OverflowError:
                    cur = cur_loss
                try:
                    elp = elapsed * 1000 / args.log_interval
                except OverflowError:
                    elp = elapsed
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f} | acc {:05.5f} | tag_acc {:05.5f}'.format(
                    epoch, batch, len(pii), optimizer.param_groups[0]['lr'],
                    elp, cur_loss, ex, cur, cur_acc, cur_tag_acc))
                total_loss = 0
                total_acc = 0
                total_tag_acc = 0
                notag_batches = 0
                start_time = time.time()
            ###
            batch += 1
            # i += seq_len

    # Loop over epochs.
    lr = args.lr
    best_val_loss = []
    stored_loss = 100000000
    


    # At any point you can hit Ctrl + C to break out of training early.
    try:
        optimizer = None
        # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(params, lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'lamb':
            from pytorch_lamb import Lamb
            optimizer = Lamb(params, lr=args.lr, weight_decay=args.wdecay, min_trust=0.25)
            #optimizer = Lamb(params, lr=args.lr, weight_decay=args.wdecay, min_trust=0.1)
            #optimizer = Lamb(params, lr=args.lr, weight_decay=args.wdecay, min_trust=0, random_min_trust=0.2, random_trust_dice=10)
            #optimizer = Lamb(params, lr=args.lr, weight_decay=args.wdecay, min_trust=0.2, random_min_trust=0.5, random_trust_dice=4)
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=1, eta_min=lr_min)

        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
                
        from lookahead import Lookahead
        if False:
            k, alpha = 5, 0.8
            print('Lookahead - k {} and alpha {}'.format(k, alpha))
            optimizer = Lookahead(base_optimizer=optimizer, k=k, alpha=alpha)

        # no amp
        # from torch.cuda import amp # changed from apex to torch.cuda
        # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        
        #model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(epoch - 1)
            # model_save(f"lastest_{args.save}")
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()

                val_loss2, val_acc2, val_tag_acc2 = evaluate()
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f} | valid acc {:05.5f} | valid tag acc {:05.5f}'.format(
                        epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2), val_acc2, val_tag_acc2))
                print('-' * 89)

                if val_loss2 < stored_loss:
                    model_save(args.save)
                    print('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()

            else:
                val_loss, val_acc, val_tag_acc = evaluate()
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f} | valid acc {:05.5f} | valid tag acc {:05.5f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2), val_acc, val_tag_acc))
                print('-' * 89)

                if val_loss < stored_loss:
                    model_save(args.save)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss

                if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                    print('Switching to ASGD')
                    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

                if epoch in args.when:
                    print('Saving model before learning rate decreased')
                    model_save('{}.e{}'.format(args.save, epoch))
                    print('Dividing learning rate by 10')
                    optimizer.param_groups[0]['lr'] /= 10.

                best_val_loss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    # model_load(args.save)
    #
    # params = list(model.parameters()) + list(criterion.parameters())
    # total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    # print('Model total parameters:', total_params)


    # Run on test data.
    # test_loss = evaluate(test_data, test_batch_size)
    # print('=' * 89)
    # print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    #     test_loss, math.exp(test_loss), test_loss / math.log(2)))
    # print('=' * 89)
