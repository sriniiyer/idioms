import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

def bottle(v):
    return v.view(-1, v.size(2)) if v is not None else None

def unbottle(v, batch_size):
    return v.view(batch_size, -1, v.size(1))

def shiftLeft(t, pad):
  shifted_t = t[:, 1:] # first dim is batch
  padding =  torch.zeros(t.size(0), 1).fill_(pad).long().cuda()
  return torch.cat((shifted_t, padding), 1)

def sort_batch_by_length(tensor: torch.autograd.Variable,
                         sequence_lengths: torch.autograd.Variable):
    """
    Sort a batch first tensor by some specified lengths.

    Parameters
    ----------
    tensor : Variable(torch.FloatTensor), required.
        A batch first Pytorch tensor.
    sequence_lengths : Variable(torch.LongTensor), required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.

    Returns
    -------
    sorted_tensor : Variable(torch.FloatTensor)
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : Variable(torch.LongTensor)
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : Variable(torch.LongTensor)
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    permuation_index : Variable(torch.LongTensor)
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """

    if not isinstance(tensor, torch.autograd.Variable) or not isinstance(sequence_lengths, torch.autograd.Variable):
        raise ConfigurationError("Both the tensor and sequence lengths must be torch.autograd.Variables.")

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)

    # This is ugly, but required - we are creating a new variable at runtime, so we
    # must ensure it has the correct CUDA vs non-CUDA type. We do this by cloning and
    # refilling one of the inputs to the function.
    index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths)))
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    index_range = torch.autograd.Variable(index_range.long())
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index

class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """
    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


class ProperLSTM(nn.LSTM):

  def forward(self, seq, seq_lens):

    if not self.batch_first:
      raise ConfigurationError("Our encoder semantics assumes batch is always first!")

    non_zero_length_mask = seq_lens.ne(0).float()
    # make zero lengths into length=1
    seq_lens = seq_lens + seq_lens.eq(0).float()

    sorted_inputs, sorted_sequence_lengths, restoration_indices, sorting_indices =\
                sort_batch_by_length(seq, seq_lens)

    packed_input = pack(sorted_inputs, sorted_sequence_lengths.data.long().tolist(), batch_first=True)
    outputs, final_states = super(ProperLSTM, self).forward(packed_input)

    unpacked_sequence, _ = unpack(outputs, batch_first=True)
    outputs = unpacked_sequence.index_select(0, restoration_indices)
    new_unsorted_states = [self.fix_hidden(state.index_select(1, restoration_indices))
                                                          for state in final_states]

    # To deal with zero length inputs
    outputs = outputs * non_zero_length_mask.view(-1, 1, 1).expand_as(outputs)
    new_unsorted_states[0] = new_unsorted_states[0] * non_zero_length_mask.view(1, -1, 1).expand_as(new_unsorted_states[0])
    new_unsorted_states[1] = new_unsorted_states[1] * non_zero_length_mask.view(1, -1, 1).expand_as(new_unsorted_states[1])

    return outputs, new_unsorted_states

  def fix_hidden(self, h):
    # (layers*directions) x batch x dim to layers x batch x (directions*dim))
    if self.bidirectional:
      h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h

class Bottle(nn.Module):
        def forward(self, input):
            if len(input.size()) <= 2:
                return super(Bottle, self).forward(input)
            size = input.size()[:2]
            out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
            return out.contiguous().view(size[0], size[1], -1)

class BottleLinear(Bottle, nn.Linear):
    pass

class BottleEmb(nn.Module):
  def forward(self, input):
    size = input.size()
    if len(size) <= 2:
      return super(BottleEmb, self).forward(input)
    if len(size) == 3:
      out = super(BottleEmb, self).forward(input.view(size[0]*size[1], -1))
      return out.contiguous().view(size[0], size[1], size[2], -1)
    elif len(size) == 4:
      out = super(BottleEmb, self).forward(input.view(size[0]*size[1]*size[2], -1))
      return out.contiguous().view(size[0], size[1], size[2], size[3], -1)

class BottleLSTMHelper(nn.Module):
  def forward(self, input, lengths):
    size = input.size()
    if len(size) <= 3:
      return super(BottleLSTMHelper, self).forward(input, lengths)
    if len(size) == 4:
      out = super(BottleLSTMHelper, self).forward(input.view(size[0]*size[1], size[2], -1), lengths.view(-1))
      return (out[0].contiguous().view(size[0], size[1], size[2], -1),
              (out[1][0].contiguous().view(out[1][0].size(0), size[0], size[1], -1),
              out[1][1].contiguous().view(out[1][1].size(0), size[0], size[1], -1))
              )

class BottleLSTM(BottleLSTMHelper, ProperLSTM):
    pass

class BottleEmbedding(BottleEmb, nn.Embedding):
    pass
