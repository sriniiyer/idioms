import torch
from torch import nn
from torch.autograd import Variable
from UtilClass import ProperLSTM
from UtilClass import BottleEmbedding, BottleLSTM

class ConcodeEncoder(nn.Module):

  def __init__(self, vocabs, opt):
    super(ConcodeEncoder, self).__init__()

    self.opt = opt
    self.vocabs = vocabs

    self.names_embedding = BottleEmbedding(
      len(vocabs['names_combined']),
      self.opt.src_word_vec_size * 2,
      padding_idx=self.vocabs['names_combined'].stoi['<blank>'])

    hidden_size  = (self.opt.decoder_rnn_size // 2 if self.opt.brnn else self.opt.rnn_size)
    self.var_rnn = BottleLSTM(
        input_size=self.opt.src_word_vec_size * 2,
        hidden_size=hidden_size,
        num_layers=self.opt.enc_layers,
        dropout=self.opt.dropout,
        bidirectional=self.opt.brnn,
        batch_first=True)

  def processNames(self, batch_names):
    varCamel = Variable(batch_names.transpose(1, 2).contiguous().cuda(), requires_grad=False)
    varCamel_lengths = varCamel.ne(self.vocabs['names_combined'].stoi['<blank>']).float().sum(2)
    varNames_camel_embeddings = self.names_embedding(varCamel) # bs x vlen x camel_len x 1024. Average on the third dimension
    var_lengths = varCamel_lengths.clone()# Just length of var .. not including camel
    var_lengths[var_lengths > 0] = 1 # Just length of var .. not including camel

    # Average the embeddings
    varCamel_lengths[varCamel_lengths==0] = 1
    varCamel_encoded = [varNames_camel_embeddings.sum(2) / varCamel_lengths.unsqueeze(2).expand(varCamel_lengths.size(0), varCamel_lengths.size(1), varNames_camel_embeddings.size(3)) , None]
    varCamel_encoded[0] = varCamel_encoded[0].unsqueeze(0) # Add an extra dimension to make it look like an rnn was used, where the first dimension would be the layer number

    return varCamel_encoded[0][-1].unsqueeze(2), var_lengths

  def processVars(self, batch_names, batch_types):

    varCamel_encoded, _ = self.processNames(batch_names)

    varTypesCamel_encoded, var_lengths = self.processNames(batch_types)
    var_input = torch.cat((varTypesCamel_encoded, varCamel_encoded), 2) # Use h of the last layer

    var_context, var_hidden = self.var_rnn(var_input, var_lengths * 2)

    var_context = var_context.view(var_context.size(0), -1, var_context.size(3)) # interleave type and name, type first
    batch_size = batch_types.size(0)
    max_var_len = batch_types.size(-1)
    var_attention_mask = Variable(var_lengths.ne(0).data.unsqueeze(2).expand(batch_size, max_var_len, 2).contiguous().view(batch_size, -1).cuda(), requires_grad=False)

    return var_context, var_attention_mask

  def processSrc(self, batch_names):
    src = Variable(batch_names.cuda(), requires_grad=False)
    src_embeddings = self.names_embedding(src)
    lengths = src.ne(self.vocabs['names_combined'].stoi['<blank>']).float().sum(1)
    self.n_src_words = lengths.sum().item()
    context, enc_hidden = self.var_rnn(src_embeddings, lengths)

    src_context = context
    src_attention_mask = Variable(batch_names.ne(self.vocabs['names_combined'].stoi['<blank>']).cuda(), requires_grad=False)
    return src_context, src_attention_mask, enc_hidden

  def forward(self, batch):
    src_context, src_attention_mask, enc_hidden = self.processSrc(batch['src'])

    ret_context = [src_context]
    ret_mask = [src_attention_mask]

    var_context, var_attention_mask = self.processVars(batch['varNames'], batch['varTypes'])
    ret_context.append(var_context)
    ret_mask.append(var_attention_mask)

    method_context, method_attention_mask = self.processVars(batch['methodNames'], batch['methodReturns'])
    ret_context.append(method_context)
    ret_mask.append(method_attention_mask)

    return tuple(ret_context), tuple(ret_mask), enc_hidden
