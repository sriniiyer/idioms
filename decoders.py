import torch
from torch.autograd import Variable
from tools.exact_unordered import order

class DecoderState():
    """ Input feed is ignored for ProdDecoder"""
    def __init__(self, rnnstate, input_feed):
      self.hidden = rnnstate
      self.input_feed = input_feed
      self.batch_size = rnnstate[0].size(0)
      self.rnn_size = rnnstate[0].size(2)

    def clone(self):
      return DecoderState((self.hidden[0].clone(), self.hidden[1].clone()), self.input_feed.clone() if self.input_feed is not None else None)

    def update_state(self, rnnstate, input_feed):
      self.hidden = rnnstate
      self.input_feed = input_feed

    def repeat_beam_size_times(self, beam_size):
      """ Repeat beam_size times along batch dimension. """
      # Vars contains h, c, and input feed. Separate it later
      self.hidden = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                               for e in self.hidden]
      self.input_feed = Variable(self.input_feed.data.repeat(beam_size, 1, 1), volatile=True)

    def beam_update(self, positions, beam_size):
      """ Update when beam advances. """
      for e in self.hidden:
        a, br, d = e.size()
        # split batch x beam into two separate dimensions
        # in order to pick the particular beam that
        # we want to update
        # Choose beam number idx
        e.data.copy_(
            e.data.index_select(1, positions))

      br, a, d = self.input_feed.size() 
      self.input_feed.data.copy_(
          self.input_feed.data.index_select(0, positions))

class Prediction():
  def __init__(self, goldNl, goldRules, goldCode, prediction, prob, attn, dataset):
    self.goldNl = goldNl
    self.goldCode = goldCode
    self.goldRules = goldRules
    self.prediction = prediction
    self.prob = prob
    self.attn = attn
    self.dataset = dataset # Different datasets need different outputs for debugging

  def output(self, prefix, idx, attention=False):
    out_file = open(prefix, 'a')
    debug_file = open(prefix + '.html', 'a')
    prob_file = open(prefix + '.probs', 'a')

    if attention:
        attention_folder = prefix+ '_attentions'
        try:
          os.makedirs(attention_folder)
        except:
          pass

        attention_img(
          self.goldNl,
          self.prediction,
          self.prediction,
          torch.stack(self.attn),
          attention_folder  + '/' + str(idx) + '.png')

    out_file.write(' '.join(self.prediction) + '\n')

    prob_file.write(' '.join(self.prediction) + '\t' + str(self.prob) + '\n')
    prob_file.close()

    debug_file.write('<b>Id:</b>' + str(idx) + '<br>')
    debug_file.write('<b>Language:</b>' + '<br>')
    debug_file.write(' '.join(self.goldNl) + '<br>')
    debug_file.write('<b>Code:</b>' + '<br>')
    debug_file.write(' '.join(self.goldCode).replace('<s>', '<b>').replace('</s>', '</b>') + '<br>')
    debug_file.write('<b>Prediction:</b>' + '<br>')
    if self.dataset == "concode":
      pred_color = "black" if self.prediction == self.goldCode[1:-1] else "red"
    else: # order insensitive match for conjunctions and disjunctions
      pred_color = "black" if order(self.prediction, 'and') == order(self.goldCode[1:-1], 'and') else "red"
    debug_file.write('<span style="color: {}">{}</span><br>'.format(pred_color,' '.join(self.prediction)))
    debug_file.write('<b>Rules:</b>' + '<br>')
    for rule in self.goldRules:
      debug_file.write('<span style="color: {}">{}</span><br>'.format("blue" if "concode_idiom" in rule else "black", rule))
    debug_file.write('<b>Probability:</b>' + '<br>')
    debug_file.write('{}<br>'.format(self.prob))

    out_file.close()
    debug_file.close()
