import torch 
from preprocess import lhs, rhs, CDDataset
from torch.autograd import Variable
import copy

class TreeBeam(object):
    def __init__(self, size, cuda, vocabs, rnn_size):

        self.size = size
        self.vocabs = vocabs
        self.tt = torch.cuda if cuda else torch
        self.rnn_size = rnn_size

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []
        assert(self.vocabs['dataset'] != '')
        self.start_symbol = {"atis": "LogicalForm_NT", "concode": "MemberDeclaration_NT", "sql": "Statement_NT"}[self.vocabs['dataset']]

        # The outputs at each time-step.
        # Start with one element
        self.nextNts = [[-1] * self.size]
        self.nextYs = [self.tt.LongTensor(self.size)
                       .fill_(self.vocabs['next_rules'].stoi['<blank>'])]
        self.nextYs[0][0] = self.vocabs['prev_rules'].stoi['<s>']
        # This is ok. The first inp is filled in from the stack.
        # and the nt is decided to be <s> based on len(prevks) == 0

        # Has EOS topped the beam yet.
        self.eosTop = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.stacks = [[(self.start_symbol, '<s>', 0, Variable(self.tt.FloatTensor(1, 1, self.rnn_size).zero_(), requires_grad=False))] for i in range(0, self.size)] # stacks for non terminals

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        # We need to return a batch here
        # the batch should contain nt, prev_rule, parent_rule, parent_states
        batch = {
          'nt' : self.tt.LongTensor(self.size, 1),
          'prev_rules': self.tt.LongTensor(self.size , 1),
          'prev_rules_split': self.tt.LongTensor(self.size , 200, 1).fill_(self.vocabs['nt'].stoi['<blank>']), # has to be padded
          'parent_rules': self.tt.LongTensor(self.size, 1),
          'parentpos': self.tt.LongTensor(self.size, 1),
          'parent_rules_split': self.tt.LongTensor(self.size, 200, 1).fill_(self.vocabs['nt'].stoi['<blank>']),
          'parent_states': {}
        }

        max_prev_rules_split = 0
        max_parent_rules_split = 0
        for i in range(0, len(self.nextYs[-1])): # this is over the beam

          # Here, we are taking the rule that was best in the previous step, and converting it into a prev_rule
          # for the next decoding step
          if len(self.prevKs) == 0: # In the beginning
            prev_rule = '<s>'
          elif self.nextYs[-1][i] >= len(self.vocabs['next_rules']): # The best Y is a copy operation. How do we convert a copy operation into a prev_rule.
            prevNt = self.vocabs['nt'].itos[self.nextNts[-1][i]]
            # What happens if prevNt is not one of the valid NTs that can generate a copy
            prev_rule = CDDataset._unk_rule_from_Nt(prevNt)
          else:
            prev_rule = self.vocabs['next_rules'].itos[self.nextYs[-1][i]]

          try:
            str_prev_rule = CDDataset.getAnonRule(prev_rule)
            prev_rule_str_splits = [str_prev_rule] if "-->" not in str_prev_rule else ([lhs(str_prev_rule)] + ['<sep>'] + rhs(str_prev_rule).split('___'))
            if len(prev_rule_str_splits) > max_prev_rules_split:
              max_prev_rules_split = len(prev_rule_str_splits)
            for k in range(0, len(prev_rule_str_splits)):
              batch['prev_rules_split'][i][k][0] = self.vocabs['nt'].stoi[prev_rule_str_splits[k]]
            batch['prev_rules'][i][0] = self.vocabs['prev_rules'].stoi[str_prev_rule]
          except:
            import ipdb; ipdb.set_trace()


          # if the stack is empty put a placeholder
          if len(self.stacks[i]) == 0:
            (nt, parent_rule, parent_pos, parent_state) = (self.start_symbol, '<s>', 0, Variable(self.tt.FloatTensor(1, 1, self.rnn_size).zero_(), requires_grad=False))
          else:
            (nt, parent_rule, parent_pos, parent_state) = self.stacks[i][-1] #.top()

          batch['parent_rules'][i][0] = self.vocabs['prev_rules'].stoi[parent_rule]
          parent_rule_str_splits = [parent_rule] if "-->" not in parent_rule else ([lhs(parent_rule)] + ['<sep>'] + rhs(parent_rule).split('___'))
          if len(parent_rule_str_splits) > max_parent_rules_split:
            max_parent_rules_split = len(parent_rule_str_splits)
          for k in range(0, len(parent_rule_str_splits)):
            batch['parent_rules_split'][i][k][0] = self.vocabs['nt'].stoi[parent_rule_str_splits[k]]

          try:
            batch['nt'][i][0] = self.vocabs['nt'].stoi[nt]
            batch['parentpos'][i][0] = parent_pos
          except:
            import ipdb; ipdb.set_trace()

          batch['parent_states'][i] = {}
          batch['parent_states'][i][0] = parent_state

        # lstm doesnt like a batch with unnecessary extra lengths. The batch should be as long as the longest sequence only, not longer
        batch['parent_rules_split'] =  batch['parent_rules_split'][:, :max_parent_rules_split,].contiguous()
        batch['prev_rules_split'] =  batch['prev_rules_split'][:, :max_prev_rules_split,].contiguous()
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, attnOut, rnn_output, inp):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if len(self.stacks[i]) == 0:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        oldStacks = self.stacks
        self.stacks = [[] for i in range(0, self.size)] # stacks for non terminals

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        self.nextNts.append([])
        for i in range(0, self.size):
          self.nextNts[-1].append(inp['nt'][self.prevKs[-1][i]][0])

        def copyStack(stacks):
          return [(copy.deepcopy(stack[0]), copy.deepcopy(stack[1]), copy.deepcopy(stack[2]), stack[3].clone()) for stack in stacks]

        self.attn.append(attnOut.index_select(0, prevK))
        self.stacks = [copyStack(oldStacks[k]) for k in prevK]
        for i in range(0, self.size):
          currentRule = (bestScoresId[i] - prevK[i] * numWords) 

          try:
            self.stacks[i].pop() # This rule has been processed. This should not error out
          except:
            # This can error out if there are very few options for the previous rules (rest are -inf) and a stack with 1e-20 is also chosen in topk
            pass

          # currentRule can be a copy index. We need the non-terminal to determine
          # which unk it is
          if currentRule < len(self.vocabs['next_rules']):
            rule = self.vocabs['next_rules'].itos[currentRule] 

            # If its a terminal rule, we dont needs its parents anymore
            if not CDDataset._is_terminal_rule(rule) and rule != '<blank>':
              # in the beginning, MemberDeclaration has only 2 options
              # so the third best in the beam is -inf
              # it should get eliminated later because the score is -inf
              rhs_split = rhs(rule).split('___')
              for idx, elem in enumerate(rhs_split[::-1]): # reverse it
                if elem.endswith('_NT'):
                  pos = 2 + len(rhs_split) - idx - 1
                  self.stacks[i].append((elem, rule, pos, rnn_output[prevK[i]].unsqueeze(0)))
          else:
            pass


        for i in range(self.nextYs[-1].size(0)):
            if len(self.stacks[i]) == 0:
                s = self.scores[i]
                if s != float('-inf'): # This can happen in the first step, when the first rule only has 2 legitimate following rules, resulting in the third being inf
                  self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if len(self.stacks[0]) == 0:
            self.eosTop = True


    def done(self):
        return self.eosTop and len(self.finished) >= 1

    def getFinal(self):
      if len(self.finished) == 0:
        self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
      self.finished.sort(key=lambda a: -a[0])
      return self.finished[0]

    def getHyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        timestep usually points to the 0-based last step on nextYs. 
        So the first element to retrieve is nextYs[timestep]
        """
        nts, hyp, attn = [], [], []
        # The size of prevK is one less than that of nextYs. So thats why,  the j+1 th index of nextYs correcspond to the jth index of prevK
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1): # This is starting one step before the last step
            nts.append(self.nextNts[j + 1][k])
            hyp.append(self.nextYs[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]
        return hyp[::-1], torch.stack(attn[::-1]), nts[::-1]



class Beam(object):
    def __init__(self, size, cuda, vocab):

        self.size = size
        self.vocab = vocab
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(self.vocab.stoi['<blank>'])]
        self.nextYs[0][0] = self.vocab.stoi['<s>']

        # Has EOS topped the beam yet.
        self._eos = self.vocab.stoi['</s>']
        self.eosTop = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = {
          'code' : self.tt.LongTensor(self.nextYs[-1]).view(-1, 1),
        }

        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, attnOut):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))
        self.attn.append(attnOut.index_select(0, prevK))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self.vocab.stoi['</s>']:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= 1

    def getFinal(self):
      if len(self.finished) == 0:
        self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
      self.finished.sort(key=lambda a: -a[0])
      return self.finished[0]

    def getHyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]
        return hyp[::-1], torch.stack(attn[::-1])
