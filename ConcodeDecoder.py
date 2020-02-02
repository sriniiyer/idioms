import torch
from torch import nn
from GlobalAttention import GlobalAttention
from torch.autograd import Variable
from Beam import TreeBeam
from UtilClass import bottle, unbottle, BottleEmbedding, BottleLSTM
from preprocess import lhs, rhs, CDDataset
from decoders import DecoderState, Prediction

class ConcodeDecoder(nn.Module):

  def __init__(self, vocabs, opt):
    super(ConcodeDecoder, self).__init__()

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
        opt.decoder_rnn_size,
        attn_type=opt.attn_type,
        include_rnn=False,
          bias=opt.attbias)

    self.attn_linear = nn.Linear(self.opt.decoder_rnn_size * 3, self.opt.decoder_rnn_size)

    self.var_attn = GlobalAttention(
        opt.decoder_rnn_size,
        attn_type=opt.attn_type,
        include_rnn=False,
          bias=opt.attbias)

    #-----------------------------------
    #Not used
    #-----------------------------------
    self.copy_attn = GlobalAttention(
      opt.decoder_rnn_size,
      attn_type=opt.attn_type,
      bias=opt.attbias)

    self.prev_rnn = BottleLSTM(
      input_size=opt.tgt_word_vec_size,
      hidden_size=opt.rnn_size // opt.prev_divider // 2,
      num_layers=1,
      dropout=opt.dropout,
      bidirectional=True,
      batch_first=True)
    #-----------------------------------

    self.decoder_rnn = nn.LSTM(
      input_size=opt.tgt_word_vec_size * 3 + opt.decoder_rnn_size, # nt and prev_rule
      hidden_size=opt.decoder_rnn_size,
      num_layers=opt.dec_layers,
      dropout=opt.dropout,
      batch_first=True)

    self.decoder_dropout = nn.Dropout(opt.dropout)

  def forward(self, batch, all_context, context_masks, decState):

    src_context = all_context[0]
    src_context_mask = context_masks[0]
    rest_context = torch.cat(all_context[1:], 1)
    rest_context_mask = torch.cat(context_masks[1:], 1)

    context = torch.cat(all_context, 1)
    context_lengths = torch.cat(context_masks, 1)

    # Embed everything
    nt_embeddings = self.nt_embedding(Variable(batch['nt'].cuda(), requires_grad=False))
    rule_embeddings = self.rule_embedding(Variable(batch['prev_rules'].cuda(), requires_grad=False))
    parent_rule_embeddings = self.rule_embedding(Variable(batch['parent_rules'].cuda(), requires_grad=False))

    attn_outputs, attn_scores, copy_attn_scores = [], [], []
    # For each batch we have to maintain states

    batch_size = batch['nt'].size(0) # 1 for predict
    num_decodes = 0

    for i, (nt, rule, parent_rule) in enumerate(zip(nt_embeddings.split(1, 1), rule_embeddings.split(1, 1), parent_rule_embeddings.split(1, 1))):
      # accumulate parent decoder states
      parent_states = []
      for j in range(0, batch_size):
        try: # this is needed coz the batch is of different sizes
          parent_states.append(batch['parent_states'][j][i]) # one state for every batch
        except:
          parent_states.append(batch['parent_states'][j][0]) # one state for every batch
      parent_states = torch.cat(parent_states, 0)

      rnn_output, prev_hidden = self.decoder_rnn(torch.cat((nt, rule, parent_rule, parent_states), 2), decState.hidden)
      num_decodes += 1
      rnn_output.contiguous()

      src_attn_output, src_attn_score = self.attn(rnn_output, src_context, src_context_mask)
      varmet_attn_output, varmet_attn_score = self.var_attn(src_attn_output, rest_context, rest_context_mask)

      attn_output = torch.tanh(self.attn_linear(torch.cat((rnn_output, src_attn_output, varmet_attn_output), 2)))
      attn_scores.append(varmet_attn_score)
      copy_attn_scores.append(varmet_attn_score)

      attn_output = self.decoder_dropout(attn_output)
      attn_outputs.append(attn_output)

      decState.update_state(prev_hidden, attn_output)

      # update all children
      for j, elem in enumerate(rnn_output.split(1, 0)):
        # children wont be there during prediction
        if 'children' in batch and i in batch['children'][j]: # rule i has children
          for child in batch['children'][j][i]:
            batch['parent_states'][j][child] = elem

    output = torch.cat(attn_outputs, 1)
    attn_scores = torch.cat(attn_scores, 1)
    copy_attn_scores = torch.cat(copy_attn_scores, 1)

    return output, attn_scores, copy_attn_scores

  def predict(self, enc_hidden, context, context_lengths, batch, beam_size, max_code_length, generator, replace_unk):

    # This decoder does not have input feeding. Parent state replaces that
    decState = DecoderState(
      enc_hidden,
      Variable(torch.zeros(1, 1, self.opt.decoder_rnn_size).cuda(), requires_grad=False) # parent state
    )
    # Repeat everything beam_size times.
    def rvar(a, beam_size):
      return Variable(a.repeat(beam_size, 1, 1), volatile=True)

    context = tuple(rvar(context[i].data, beam_size) for i in range(0, len(context)))
    context_lengths = tuple(context_lengths[i].repeat(beam_size, 1) for i in range(0, len(context_lengths)))

    decState.repeat_beam_size_times(beam_size) # TODO: get back to this

    # Use only one beam
    beam = TreeBeam(beam_size, True, self.vocabs, self.opt.decoder_rnn_size)

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
        src_map = torch.cat((batch['concode_src_map_vars'], batch['concode_src_map_methods']), 1)

        scores = generator(bottle(output), bottle(copy_attn), src_map, inp) #generator needs the non-terminals

        # One for every beam, to make it look like batch while training
        inp['concode_vocab'] = {}
        for bs in range(0, beam_size):
          inp['concode_vocab'][bs] = batch['concode_vocab'] if 'concode_vocab' in batch else None

        out = generator.collapseCopyScores(unbottle(scores.data.clone(), beam_size), inp) # needs seq2seq from batch
        out = out.log()

        # beam x tgt_vocab

        beam.advance(out[:, 0],  attn.data[:, 0], output, inp)
        decState.beam_update(beam.getCurrentOrigin(), beam_size)

    score, times, k = beam.getFinal() # times is the length of the prediction
    hyp, att, nts = beam.getHyp(times, k)
    goldNl = []
    goldNl += batch['concode_var'][0] # because batch = 1
    goldNl += batch['concode_method'][0] # because batch = 1

    goldCode = self.vocabs['code'].addStartOrEnd(batch['raw_code'][0])
    predRules, copied_tokens, replaced_tokens = self.buildTargetRules(
      hyp,
      nts,
      self.vocabs,
      goldNl,
      att,
      batch['concode_vocab'][0],
      replace_unk
    )
    predSent = ConcodeDecoder.rulesToCode(predRules)

    return Prediction(batch['raw_seq2seq'][0], predRules, goldCode, predSent, score, att, self.vocabs['dataset'])


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
      tokens = []
      copied_tokens, replaced_tokens = [], []
      for j, tok in enumerate(pred):
          if tok < len(vocab):
              rules.append(vocab.itos[tok])
          else:
            try:
              nt = vocabs['nt'].itos[nts[j]]
              rules.append(nt + "-->" + copy_vocab.itos[tok - len(vocab)])
              copied_tokens.append(copy_vocab.itos[tok - len(vocab)])
            except: # The previous rule was not one with a valid NT. Ignore this rule
              pass

      if replace_unk and attn is not None:
          for i in range(len(rules)):
              if CDDataset._unk_rule(rules[i]):
                  _, maxIndex = attn[i].max(0)
                  rules[i] = lhs(rules[i]) + "-->" + src[maxIndex.item()]
                  replaced_tokens.append(src[maxIndex.item()])

      return rules, copied_tokens, replaced_tokens
