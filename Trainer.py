from Statistics import Statistics
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn

class Trainer:

  def __init__(self, model):
    self.model = model
    self.opt = model.opt
    self.start_epoch = self.opt.start_epoch if self.opt.start_epoch  else 1

    self.lr = self.opt.learning_rate
    self.betas = [0.9, 0.98]
    self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
                                betas=self.betas, eps=1e-9)

    if 'prev_optim' in self.opt:
      print('Loading prev optimizer state')
      self.optimizer.load_state_dict(self.opt.prev_optim)
      for state in self.optimizer.state.values():
        for k, v in state.items():
          if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

    if self.opt.param_init > 0:
      print('Initializating params with uniform: {}'.format(self.opt.param_init))
      for p in self.model.parameters():
        if p.requires_grad:
          p.data.uniform_(-self.opt.param_init, self.opt.param_init)

      # Restore embedding padding indexes
      try:
        self.model.encoder.encoder_embedding.weight.data[self.model.encoder.encoder_embedding.padding_idx] = 0
      except:
        pass
      try:
        self.model.decoder.decoder_embedding.weight.data[self.model.decoder.decoder_embedding.padding_idx] = 0
      except:
        pass
      try:
        self.model.decoder.nt_embedding.weight.data[self.model.decoder.nt_embedding.padding_idx] = 0
      except:
        pass
      try:
        self.model.decoder.rule_embedding.weight.data[self.model.decoder.rule_embedding.padding_idx] = 0
      except:
        pass

  def save_checkpoint(self, epoch, valid_stats):
      real_model = self.model

      model_state_dict = real_model.state_dict()
      self.opt.learning_rate = self.lr
      checkpoint = {
          'model': model_state_dict,
          'vocab': real_model.vocabs,
          'opt':   self.opt,
          'epoch': epoch,
          'optim': self.optimizer.state_dict()
        }
      torch.save(checkpoint,
                 '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                 % (self.opt.save_model, valid_stats.accuracy(),
                    valid_stats.ppl(), epoch))

  def update_learning_rate(self, valid_stats):
    if self.last_ppl is not None and valid_stats.ppl() > self.last_ppl:
        self.lr = self.lr * self.opt.learning_rate_decay
        print("Decaying learning rate to %g" % self.lr)

    self.last_ppl = valid_stats.ppl()
    self.optimizer.param_groups[0]['lr'] = self.lr

  def run_train_batched(self, train_data, valid_data, vocabs):
    print(self.model.parameters)

    total_train = train_data.compute_batches(self.opt.batch_size, vocabs, self.opt.max_chars, 0, 1, self.opt.decoder_type,  trunc=self.opt.trunc)
    total_valid = valid_data.compute_batches(self.opt.batch_size, vocabs, self.opt.max_chars, 0, 1, self.opt.decoder_type, randomize=False, trunc=self.opt.trunc)

    print('Computed Batches. Total train={}, Total valid={}'.format(total_train, total_valid))

    report_stats = Statistics()
    self.last_ppl = None

    for epoch in range(self.start_epoch, self.opt.epochs + 1):
      self.model.train()

      total_stats = Statistics()
      batch_number = -1

      for idx, batch in enumerate(train_data.batches):
        batch['gpu'] = self.opt.gpuid[0]
        loss, batch_stats = self.model.forward(batch)
        batch_size = batch['code'].size(0)
        loss.div(batch_size).backward() 
        report_stats.update(batch_stats)
        total_stats.update(batch_stats)
        batch_number += 1

        clip_grad_norm_(self.model.parameters(), self.opt.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        if batch_number % self.opt.report_every == -1 % self.opt.report_every:
          report_stats.output(epoch, batch_number + 1, len(train_data.batches), total_stats.start_time)
          report_stats = Statistics()

      print('Train perplexity: %g' % total_stats.ppl())
      print('Train accuracy: %g' % total_stats.accuracy())

      self.model.eval()
      valid_stats = Statistics()
      for idx, batch in enumerate(valid_data.batches):
        batch['gpu'] = self.opt.gpuid[0]
        loss, batch_stats = self.model.forward(batch)
        valid_stats.update(batch_stats)

      print('Validation perplexity: %g' % valid_stats.ppl())
      print('Validation accuracy: %g' % valid_stats.accuracy())

      self.update_learning_rate(valid_stats)
      print('Saving model')
      self.save_checkpoint(epoch, valid_stats)
      print('Model saved')
