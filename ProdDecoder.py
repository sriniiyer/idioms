import torch
from torch import nn
from GlobalAttention import GlobalAttention
from torch.autograd import Variable
from Beam import TreeBeam
from UtilClass import bottle, unbottle, BottleEmbedding, BottleLSTM
from preprocess import lhs, rhs, CDDataset
from decoders import DecoderState, Prediction

class ProdDecoder(nn.Module):

  def __init__(self, vocabs, opt):
    super(ProdDecoder, self).__init__()

    self.opt = opt
    self.vocabs = vocabs

    self.nt_embedding = BottleEmbedding(
      len(vocabs['nt']),
      opt.tgt_word_vec_size,
      padding_idx=vocabs['nt'].stoi['<blank>'])

    self.rule_embedding = nn.Embedding(
      len(vocabs['prev_rules']),
      opt.tgt_word_vec_size,
      padding_idx=vocabs['prev_rules'].stoi['<blank>'])

    self.attn = GlobalAttention(
        opt.rnn_size,
        attn_type=opt.attn_type,
          bias=opt.attbias)

    self.copy_attn = GlobalAttention(
        opt.rnn_size,
        attn_type=opt.attn_type,
        bias=opt.attbias)

    self.prev_rnn = BottleLSTM(
      input_size=opt.tgt_word_vec_size,
      hidden_size=opt.rnn_size // opt.prev_divider // 2,
      num_layers=1,
      dropout=opt.dropout,
      bidirectional=True,
      batch_first=True)

    self.decoder_rnn = nn.LSTM(
      input_size=opt.tgt_word_vec_size + opt.rnn_size // opt.prev_divider + opt.rnn_size // opt.prev_divider,
      hidden_size=opt.rnn_size,
      num_layers=opt.dec_layers,
      dropout=opt.dropout,
      batch_first=True)

    self.decoder_dropout = nn.Dropout(opt.dropout)

  def forward(self, batch, context, context_lengths, decState):

    # embed everything
    nt_embeddings = self.nt_embedding(Variable(batch['nt'].cuda(), requires_grad=False))

    rule_embeddings = self.rule_embedding(Variable(batch['prev_rules'].cuda(), requires_grad=False))
    split_rule_embeddings = self.nt_embedding(Variable(batch['prev_rules_split'].transpose(1, 2).cuda(), requires_grad=False))
    split_rule_embedding_lengths = Variable(batch['prev_rules_split'].ne(self.vocabs['nt'].stoi['<blank>']).float().sum(1).cuda(), requires_grad=False)
    parent_rule_embeddings = self.rule_embedding(Variable(batch['parent_rules'].cuda(), requires_grad=False))
    parent_split_rule_embeddings = self.nt_embedding(Variable(batch['parent_rules_split'].transpose(1, 2).cuda(), requires_grad=False))
    parent_split_rule_embedding_lengths = Variable(batch['parent_rules_split'].ne(self.vocabs['nt'].stoi['<blank>']).float().sum(1).cuda(), requires_grad=False)

    split_rule_context, split_rule_hidden = self.prev_rnn(split_rule_embeddings, split_rule_embedding_lengths)
    split_parent_context, split_parent_hidden = self.prev_rnn(parent_split_rule_embeddings, parent_split_rule_embedding_lengths)

    attn_outputs, attn_scores, copy_attn_scores = [], [], []
    # For each batch we have to maintain states

    batch_size = batch['nt'].size(0) # 1 for predict
    num_decodes = 0

    attn_outputs, attn_scores, copy_attn_scores = [], [], []
    for i, (nt, rule, parent_rule, split_rule, split_parent_rule) in enumerate(zip(nt_embeddings.split(1, 1), rule_embeddings.split(1, 1), parent_rule_embeddings.split(1, 1), split_rule_hidden[0][-1].split(1, 1), split_parent_hidden[0][-1].split(1, 1))):
      # accumulate parent decoder states

      rnn_output, prev_hidden = self.decoder_rnn(torch.cat((nt, split_rule, split_parent_rule), 2), decState.hidden)

      num_decodes += 1
      rnn_output.contiguous()
      attn_output, attn_score = self.attn(rnn_output, context, context_lengths)
      attn_scores.append(attn_score)
      attn_output = self.decoder_dropout(attn_output)
      attn_outputs.append(attn_output)

      decState.update_state(prev_hidden, attn_output)
      _, copy_attn_score = self.copy_attn(attn_output, context, context_lengths)
      copy_attn_scores.append(copy_attn_score)

    output = torch.cat(attn_outputs, 1)
    attn_scores = torch.cat(attn_scores, 1)
    copy_attn_scores = torch.cat(copy_attn_scores, 1)

    return output, attn_scores, copy_attn_scores

  def predict(self, enc_hidden, context, context_lengths, batch, beam_size, max_code_length, generator, replace_unk):

    # This decoder does not have input feeding. Parent state replaces that
    decState = DecoderState(
      enc_hidden,
      Variable(torch.zeros(1, 1, self.opt.rnn_size).cuda(), requires_grad=False) # placeholder, is ignored for ProdDecoder
    )
    # Repeat everything beam_size times.
    def rvar(a, beam_size):
      return Variable(a.repeat(beam_size, 1, 1), volatile=True)
    context = rvar(context.data, beam_size)
    context_lengths = context_lengths.repeat(beam_size)
    decState.repeat_beam_size_times(beam_size) # TODO: get back to this

    # Use only one beam
    beam = TreeBeam(beam_size, True, self.vocabs, self.opt.rnn_size)

    for count in range(0, max_code_length): # We will break when we have the required number of terminals
      # to be consistent with seq2seq

      if beam.done(): # TODO: fix b.done
        break

      # Construct batch x beam_size nxt words.
      # Get all the pending current beam words and arrange for forward.
      # Uses the start symbol in the beginning
      inp = beam.getCurrentState() # Should return a batch of the frontier

      # Run one step., decState gets automatically updated
      output, attn, copy_attn = self.forward(inp, context, context_lengths, decState)
      scores = generator(bottle(output), bottle(copy_attn), batch['src_map'], inp) #generator needs the non-terminals

      # One for every beam, to make it look like batch while training
      inp['seq2seq_vocab'] = {}
      inp['concode_vocab'] = {}
      for bs in range(0, beam_size):
        inp['seq2seq_vocab'][bs] = batch['seq2seq_vocab'] if 'seq2seq_vocab' in batch else None
        inp['concode_vocab'][bs] = batch['concode_vocab'] if 'concode_vocab' in batch else None

      out = generator.collapseCopyScores(unbottle(scores.data.clone(), beam_size), inp) # needs seq2seq from batch
      out = out.log()

      # beam x tgt_vocab

      beam.advance(out[:, 0],  attn.data[:, 0], output, inp)
      decState.beam_update(beam.getCurrentOrigin(), beam_size)

    score, times, k = beam.getFinal() # times is the length of the prediction
    hyp, att, nts = beam.getHyp(times, k)
    goldNl = self.vocabs['seq2seq'].addStartOrEnd(batch['raw_seq2seq'][0]) # because batch = 1
    goldCode = self.vocabs['code'].addStartOrEnd(batch['raw_code'][0])
    predRules = self.buildTargetRules(
      hyp,
      nts,
      self.vocabs,
      goldNl,
      att,
      batch['seq2seq_vocab'][0],
      replace_unk
    )
    predSent = ProdDecoder.rulesToCode(predRules)
    return Prediction(goldNl, predRules, goldCode, predSent, score, att, self.vocabs['dataset'])


  @staticmethod
  def rulesToCode(rules):
    stack = []
    code = []
    for i in range(0, len(rules)):
      if not CDDataset._is_terminal_rule(rules[i]):
        stack.extend(rhs(rules[i]).replace('concode_idiom___', '').split('___')[::-1]) # Removing concode_idiom. We introduced this so that we could color rules in the tree in order to identify idioms.
      else:
        code.append(rhs(rules[i]))

      try:
        top = stack.pop()

        while not top.endswith('_NT'):
          code.append(top)
          if len(stack) == 0:
            break
          top = stack.pop()
      except:
        pass

    return code

  def buildTargetRules(self, pred, nts, vocabs, src, attn, copy_vocab, replace_unk):
      vocab = vocabs['next_rules']
      rules = []
      for j, tok in enumerate(pred):
        if tok < len(vocab): # Not a copy operation
            rules.append(vocab.itos[tok])
        else: # Something should be copied. So we should figure out the right non-terminal
          try:
            nt = vocabs['nt'].itos[nts[j]]
            rules.append(nt + "-->" + copy_vocab.itos[tok - len(vocab)])
          except: # The previous rule was not one with a valid NT. Ignore this rule
            pass

      if replace_unk and attn is not None:
        for i in range(len(rules)):
          if CDDataset._unk_rule(rules[i]):
            _, maxIndex = attn[i].max(0)
            rules[i] = lhs(rules[i]) + "-->" + src[maxIndex[0]]

      return rules
