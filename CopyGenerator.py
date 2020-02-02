import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from UtilClass import shiftLeft, bottle, unbottle
from preprocess import CDDataset, lhs

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)

class ProdGenerator(nn.Module):
  def __init__(self, rnn_size, vocabs, opt):
    super(ProdGenerator, self).__init__()
    self.opt = opt
    self.mask = Variable(vocabs['mask'].float().cuda(), requires_grad=False)
    self.linear = nn.Linear(rnn_size , len(vocabs['next_rules']))  # only non unk rules
    self.linear_copy = nn.Linear(rnn_size, 1)
    self.tgt_pad = vocabs['next_rules'].stoi['<blank>']

    # This now depends on the non-terminal. The UNKs are now typed
    self.tgt_unks = []

    # Add all the UNK rules in numeric form here
    for pre_terminal in CDDataset.pre_terminal_symbols:
      self.tgt_unks.append(vocabs['next_rules'].stoi[CDDataset._unk_rule_from_Nt(pre_terminal)])

    self.vocabs = vocabs

  def forward(self, hidden, attn, src_map, batch):
    out = self.linear(hidden)
    # batch['nt'] contains padding. 
    batch_size, slen_, cvocab = src_map.size()

    # batch['nt'] is bs x tgt_len
    non_terminals = batch['nt'].contiguous().cuda().view(-1)  # bottled(bs x tgt_len)
    masked_out = torch.add(out, torch.index_select(self.mask, 0, Variable(non_terminals, requires_grad=False)))
    # bottled indexes that are not blank
    prob = F.softmax(masked_out, dim=1)

    batch_by_tlen_, slen = attn.size()
    # Probability of copying p(z=1) batch.
    copy = torch.sigmoid(self.linear_copy(hidden)) # bottled(bs x tgt_len)

    # Probability of not copying: p_{word}(w) * (1 - p(z))
    for i, pre_terminal in enumerate(CDDataset.pre_terminal_symbols):
      pre_terminal_mask = non_terminals.cuda().view(-1, 1).eq(self.vocabs['nt'].stoi[pre_terminal])
      if i == 0:
        copy_enabled_nonterminals = pre_terminal_mask.clone()
      else:
        copy_enabled_nonterminals = copy_enabled_nonterminals | pre_terminal_mask

    masked_copy = Variable(copy_enabled_nonterminals.float()) * copy

    out_prob = torch.mul(prob,  1 - masked_copy.expand_as(prob)) # The ones without IdentifierNT are left untouched
    mul_attn = torch.mul(attn, masked_copy.expand_as(attn)) # Here, all non-copy are 0
    copy_prob = torch.bmm(mul_attn.view(batch_size, -1, slen), Variable(src_map.cuda(), requires_grad=False))
    # bs x tgt_len x src_sent_vocab_size
    copy_prob = copy_prob.view(-1, cvocab) # bottle it again to get batch_by_len times cvocab
    # bottled(bs x tgt_len) x src_sent_vocab_size
    return torch.cat([out_prob, copy_prob], 1) # batch_by_tlen x (out_vocab + cvocab)

  def computeLoss(self, scores, batch):

    batch_size = batch['seq2seq'].size(0)

    target = Variable(batch['next_rules'].contiguous().cuda().view(-1), requires_grad=False)
    if self.opt.decoder_type == "prod":
      align = Variable(batch['next_rules_in_src_nums'].contiguous().cuda().view(-1), requires_grad=False)
      align_unk = batch['seq2seq_vocab'][0].stoi['<unk>']
    elif self.opt.decoder_type in ["concode"]:
      align = Variable(batch['concode_next_rules_in_src_nums'].contiguous().cuda().view(-1), requires_grad=False)
      align_unk = batch['concode_vocab'][0].stoi['<unk>']

    offset = len(self.vocabs['next_rules'])

    out = scores.gather(1, align.view(-1, 1) + offset).view(-1).mul(align.ne(align_unk).float()) # all where copy is not unk
    tmp = scores.gather(1, target.view(-1, 1)).view(-1)

    unk_mask = target.data.ne(self.tgt_unks[0])
    for unk in self.tgt_unks:
      unk_mask = unk_mask & target.data.ne(unk)
    unk_mask_var = Variable(unk_mask, requires_grad=False)
    inv_unk_mask_var = Variable(~unk_mask, requires_grad=False)

    out = out + 1e-20 + tmp.mul(unk_mask_var.float()) + \
                  tmp.mul(align.eq(align_unk).float()).mul(inv_unk_mask_var.float()) # copy and target are unks

        # Drop padding.
    loss = -out.log().mul(target.ne(self.tgt_pad).float()).sum()
    scores_data = scores.data.clone()
    target_data = target.data.clone() #computeLoss populates this

    scores_data = self.collapseCopyScores(unbottle(scores_data, batch_size), batch)
    scores_data = bottle(scores_data)

    # Correct target copy token instead of <unk>
    # tgt[i] = align[i] + len(tgt_vocab)
    # for i such that tgt[i] == 0 and align[i] != 0
    # when target is <unk> but can be copied, make sure we get the copy index right
    correct_mask = inv_unk_mask_var.data * align.data.ne(align_unk)
    correct_copy = (align.data + offset) * correct_mask.long()
    target_data = (target_data * (~correct_mask).long()) + correct_copy

    pred = scores_data.max(1)[1]
    non_padding = target_data.ne(self.tgt_pad)
    num_correct = pred.eq(target_data).masked_select(non_padding).sum()

    return loss, non_padding.sum(), num_correct #, stats

  def collapseCopyScores(self, scores, batch):
    #TODO: Don't use elements from batch on the output side here. They are not available during prediction.
    """
    Given scores from an expanded dictionary
    corresponding to a batch, sums together copies,
    with a dictionary word when it is ambigious.
    """
    tgt_vocab = self.vocabs['next_rules']
    offset = len(tgt_vocab)
    for b in range(scores.size(0)): # loop over batches. Was a bug here. Used to be batch['seq2seq'].size(0) which is wrong during prediction
      if self.opt.decoder_type == "prod":
        src_vocab = batch['seq2seq_vocab'][b]
      elif self.opt.decoder_type in ["concode"]:
        src_vocab = batch['concode_vocab'][b]

      for nt in range(scores.size(1)): # Loop through every rule's NT in the decoder side
        nt_str = self.vocabs['nt'].itos[batch['nt'][b][nt]]

        if lhs(nt_str) in CDDataset.pre_terminal_symbols: # This NT can produce a copy action
          prefix = lhs(nt_str) + '-->'
        else:
          continue # This NT cannot produce a copy action

        # src_vocab is the small vocabulary just for this example
        for i in range(2, len(src_vocab)): # skip blank and unk, loop through every element
          sw = prefix + src_vocab.itos[i] # Generate the rule that makes this NT directly produce the target word, instead of copying it
          if sw in tgt_vocab.stoi: # Does this rule exist directly
            ti = tgt_vocab.stoi[sw]
            scores[b, nt, ti] += scores[b, nt, offset + i] # Yes the rule exists. Add the copy probability of this word to the prob mass of the rule to directly produce it.
            scores[b, nt, offset + i] = 1e-20
    return scores

class CopyGenerator(nn.Module):
    """
    Generator module that additionally considers copying
    words directly from the source.
    """
    def __init__(self, rnn_size, vocabs, opt):
        super(CopyGenerator, self).__init__()
        self.opt = opt
        self.tgt_dict_size = len(vocabs['code'])
        self.tgt_padding_idx = vocabs['code'].stoi['<blank>']
        self.tgt_unk_idx = vocabs['code'].stoi['<unk>']
        self.vocabs = vocabs
        self.linear = nn.Linear(rnn_size, self.tgt_dict_size)
        self.linear_copy = nn.Linear(rnn_size, 1)
        force_copy=False
        self.criterion = CopyGeneratorCriterion(self.tgt_dict_size, force_copy, self.tgt_padding_idx, self.tgt_unk_idx)

    def forward(self, hidden, copy_attn, src_map, batch):
        """
        Computes p(w) = p(z=1) p_{copy}(w|z=0)  +  p(z=0) * p_{softmax}(w|z=0)
        """
        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_size, slen_, cvocab = src_map.size()

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.tgt_padding_idx] = -float('inf')
        prob = F.softmax(logits, dim=1)

        assert(copy_attn is not None)

        batch_by_tlen_, slen = copy_attn.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)
        # Probability of copying p(z=1) batch.
        copy = torch.sigmoid(self.linear_copy(hidden))

        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob,  1 - copy.expand_as(prob))
        mul_attn = torch.mul(copy_attn, copy.expand_as(copy_attn))
        copy_prob = torch.bmm(mul_attn.view(batch_size, -1, slen), Variable(src_map.cuda(), requires_grad=False))
        copy_prob = copy_prob.view(-1, cvocab) # bottle it again to get batch_by_len times cvocab
        return torch.cat([out_prob, copy_prob], 1) # batch_by_tlen x (out_vocab + cvocab)

    def computeLoss(self, scores, batch):
        """
        Args:
            batch: the current batch.
            target: the validate target to compare output with.
            align: the align info.
        """
        batch_size = batch['seq2seq'].size(0)

        self.target = Variable(shiftLeft(batch['code'].cuda(), self.tgt_padding_idx).view(-1), requires_grad=False)

        align = Variable(shiftLeft(batch['code_in_src_nums'].cuda(),  self.vocabs['seq2seq'].stoi['<blank>']).view(-1), requires_grad=False)
        # All individual vocabs have the same unk index
        align_unk = batch['seq2seq_vocab'][0].stoi['<unk>']
        loss = self.criterion(scores, self.target, align, align_unk)

        scores_data = scores.data.clone()
        target_data = self.target.data.clone() #computeLoss populates this

        scores_data = self.collapseCopyScores(unbottle(scores_data, batch_size), batch)
        scores_data = bottle(scores_data)

        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        # when target is <unk> but can be copied, make sure we get the copy index right
        correct_mask = target_data.eq(self.tgt_unk_idx) * align.data.ne(align_unk)
        correct_copy = (align.data + self.tgt_dict_size) * correct_mask.long()
        target_data = (target_data * (1 - correct_mask).long()) + correct_copy


        pred = scores_data.max(1)[1]
        non_padding = target_data.ne(self.tgt_padding_idx)
        num_correct = pred.eq(target_data).masked_select(non_padding).sum()

        return loss, non_padding.sum(), num_correct #, stats

    def collapseCopyScores(self, scores, batch):
      """
      Given scores from an expanded dictionary
      corresponding to a batch, sums together copies,
      with a dictionary word when it is ambigious.
      """
      tgt_vocab = self.vocabs['code']
      offset = len(tgt_vocab)
      for b in range(batch['seq2seq'].size(0)):
        src_vocab = batch['seq2seq_vocab'][b]
        for i in range(1, len(src_vocab)):
          sw = src_vocab.itos[i]
          ti = tgt_vocab.stoi[sw] if sw in tgt_vocab.stoi else self.tgt_unk_idx
          if ti != self.tgt_unk_idx:
            scores[b, :, ti] += scores[b, :, offset + i]
            scores[b, :, offset + i].fill_(1e-20)
      return scores


class CopyGeneratorCriterion(object):
    def __init__(self, vocab_size, force_copy, tgt_pad, tgt_unk, eps=1e-20):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = vocab_size
        self.tgt_pad = tgt_pad
        self.tgt_unk = tgt_unk

    def __call__(self, scores, target, align, copy_unk):
        # Copy prob.
        out = scores.gather(1, align.view(-1, 1) + self.offset) \
                    .view(-1).mul(align.ne(copy_unk).float())
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)

        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            # first one = target is not unk
            out = out + self.eps + tmp.mul(target.ne(self.tgt_unk).float()) + \
                  tmp.mul(align.eq(copy_unk).float()).mul(target.eq(self.tgt_unk).float()) # copy and target are unks
        else:
            # Forced copy.
            out = out + self.eps + tmp.mul(align.eq(0).float())

        # Drop padding.
        loss = -out.log().mul(target.ne(self.tgt_pad).float()).sum()
        return loss
