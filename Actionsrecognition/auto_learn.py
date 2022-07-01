import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable


def my_softmax(input, axis=1):
	trans_input = input.transpose(axis, 0).contiguous()
	soft_max_1d = F.softmax(trans_input)
	return soft_max_1d.transpose(axis, 0)


def get_offdiag_indices(num_nodes):
	ones = torch.ones(num_nodes, num_nodes)
	eye = torch.eye(num_nodes, num_nodes)
	offdiag_indices = (ones - eye).nonzero().t()
	offdiag_indices_ = offdiag_indices[0] * num_nodes + offdiag_indices[1]
	return offdiag_indices, offdiag_indices_


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
	y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
	if hard:
		shape = logits.size()
		_, k = y_soft.data.max(-1)
		y_hard = torch.zeros(*shape)
		if y_soft.is_cuda:
			y_hard = y_hard.cuda()
		y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
		y = Variable(y_hard - y_soft.data) + y_soft
	else:
		y = y_soft
	return y


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
	gumbel_noise = sample_gumbel(logits.size(), eps=eps)
	if logits.is_cuda:
		#print(logits.get_device())
		gumbel_noise = gumbel_noise.cuda()
		#print(gumbel_noise.get_device())
		
	y = logits + Variable(gumbel_noise)
	return my_softmax(y / tau, axis=-1)


def sample_gumbel(shape, eps=1e-10):
	uniform = torch.rand(shape).float()
	return - torch.log(eps - torch.log(uniform + eps))


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


class MLP(nn.Module):

	def __init__(self, n_in, n_hid, n_out, do_prob=0.):
		super().__init__()

		self.fc1 = nn.Linear(n_in, n_hid)
		self.fc2 = nn.Linear(n_hid, n_out)
		self.bn = nn.BatchNorm1d(n_out)
		self.dropout = nn.Dropout(p=do_prob)

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight.data)
				m.bias.data.fill_(0.1)
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def batch_norm(self, inputs):
		x = inputs.view(inputs.size(0) * inputs.size(1), -1)
		x = self.bn(x)
		return x.view(inputs.size(0), inputs.size(1), -1)
		
	def forward(self, inputs):
		x = F.elu(self.fc1(inputs))
		x = self.dropout(x)
		x = F.elu(self.fc2(x))
		return self.batch_norm(x)


class InteractionNet(nn.Module):

	def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
		super().__init__()

		self.factor = factor
		self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
		self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
		self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
		self.mlp4 = MLP(n_hid*2, n_hid, n_hid, do_prob)
		self.fc_out = nn.Linear(n_hid, n_out)

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight.data)
				m.bias.data.fill_(0.1)

	
	def forward(self, inputs):            
		
		x = inputs.contiguous()
		x = x.view(inputs.size(0), inputs.size(1), -1)        
		# torch.Size([32, 14, 80])
		x = self.mlp1(x) 
		#print('mlp1',np.shape(x))        
		#  torch.Size([32, 14, 128])                            
		
		x = self.mlp2(x)                                       # [N, 600, 512] -> [N, 600, n_hid=256] -> [N, 600, n_out=256]
		# torch.Size([32, 14, 128])
		x_skip = x
		x = self.mlp3(x)
		
		x = torch.cat((x, x_skip), dim=2)
		x = self.mlp4(x)
		#print('mlp4',np.shape(x))
	
		
		
		return self.fc_out(x)                                  


class Decoder(nn.Module):
	def __init__(self, n_in, n_hid, do_prob=0):
		super().__init__()
		
		self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
		self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
		self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
		self.mlp4 = MLP(n_hid*2, n_hid, n_hid, do_prob)
		self.fc_out = nn.Linear(n_hid, 60)

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight.data)
				m.bias.data.fill_(0.1)
	def forward(self, inputs):            
		
		x = inputs.contiguous()
		#x = x.view(inputs.size(0), inputs.size(1), -1)        
		#print('decg',np.shape(x))
		# torch.Size([32, 14, 80])
		x = self.mlp1(x) 
		#print('mlp1',np.shape(x))        
		#  torch.Size([32, 14, 128])                            
		
		x = self.mlp2(x)                                       # [N, 600, 512] -> [N, 600, n_hid=256] -> [N, 600, n_out=256]
		
		x_skip = x
		x = self.mlp3(x)
		
		x = torch.cat((x, x_skip), dim=2)
		x = self.mlp4(x)
		#print('mlp4',np.shape(x))
		x = self.fc_out(x)
		
		
		return x                                 


class Autolearn(nn.Module):

	def __init__(self, n_in_enc, n_hid_enc, n_out_enc, n_in_dec, n_hid_dec, node_num=14):
		super().__init__()

		self.encoder = InteractionNet(n_in=n_in_enc,                        # 60
			                          n_hid=n_hid_enc,                      # 128
			                          n_out=n_out_enc,                     # 3
			                          do_prob=0.5,
			                          factor=True)
		self.decoder = Decoder(n_in_dec,      
			                            n_hid=n_hid_dec,         # 256
			                            do_prob=0.5,)


		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, inputs):

		N, C, T, V = inputs.size()
		x = inputs.permute(0, 3, 1, 2).contiguous()
		# N,V,C,T
		x = x.contiguous().view(N, V, C, T).permute(0,1,3,2)   ###torch.Size([32, 14, 40, 2]) (my data)
		
		hidden = self.encoder(x)   # a = torch.Size([32, 14, 10]
 		
		decoder_out = self.decoder(hidden)
		#print(',ds',np.shape(decoder_out))
		
		


		return decoder_out
