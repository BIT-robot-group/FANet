from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.msra_resnet import get_pose_net
from .networks.msra_resnet_nl import get_pose_net as get_pose_net_nonlocal
from .networks.large_hourglass import get_large_hourglass_net
from .networks.large_hourglass_nl import get_large_hourglass_nl_net
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.vggnet_nl import get_pose_net as get_vggnet_nonlocal
from .networks.alexnet_nl import get_pose_net as get_alexnet_nonlocal
from .networks.resnet_dcn import get_pose_net as get_resnetdcn
from .networks.pose_drn_aspp import get_pose_net as get_drn
# from .networks.pose_dla_dcn_nonlocal_att import get_pose_net as get_dla_dcn_nonlocal
from .networks.pose_dla_dcn_nonlocal import get_pose_net as get_dla_dcn_nonlocal
from .networks.pose_transformer_aspp import get_pose_net as get_transformer
from .networks.pose_fanet import get_pose_net as get_fanet
import warnings
warnings.filterwarnings("ignore")

_model_factory = {
  'res': get_pose_net_nonlocal,
  'hourglass': get_large_hourglass_nl_net,
  'dla': get_dla_dcn_nonlocal, #get_dla_dcn_nonlocal
  'vgg': get_vggnet_nonlocal,
  'alex': get_alexnet_nonlocal,
  'resdcn': get_resnetdcn,
  'drn': get_drn,
  'transformer': get_transformer,
  'fanet': get_fanet
}

def create_model(arch, heads, head_conv):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  
  return model

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      # if k.split('.')[0] == 'non_local':
      #   terms = k.split('.')
      #   k_r = None
      #   for idx, term in enumerate(terms):
      #     if idx == 0:
      #       k_r = term
      #     elif idx == 1:
      #       pass
      #     else:
      #       k_r = k_r + '.' + term
      #
      #   state_dict[k_r] = state_dict_[k]
      # else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}.'.format(
          k, model_state_dict[k].shape, state_dict[k].shape))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k))
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k))
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

