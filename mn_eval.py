import os
import time
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from shutil import copyfile
from tqdm import tqdm
from torch.utils import data
from torch.optim.adadelta import Adadelta
from sklearn.model_selection import train_test_split
import torch.optim as optim
from Actionsrecognition.Models import *
from Actionsrecognition.adj_learn import *
from Actionsrecognition.auto_learn import *
from collections import OrderedDict
from Visualizer import plot_graphs, plot_confusion_metrix
from utils import *
from modules import *
from torch.optim import lr_scheduler
import copy
from sklearn.metrics import confusion_matrix


device = 'cuda:0'
epochs = 10
batch_size = 32



save_folder = 'TSSTG0518'

data_files = ['../Data/scalepose/(eval)Home_new-set(labelXscrw).pkl']
#class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
#               'Stand up', 'Sit down', 'Fall Down']

class_names = ['Normal', 'Fall Down']

num_class = len(class_names)



def load_dataset(data_files, batch_size, split_size=0):
    """Load data files into torch DataLoader with/without spliting train-test.
    """
    features, labels = [], []
    for fil in data_files:
        with open(fil, 'rb') as f:
            fts, lbs = pickle.load(f)
           
            features = fts
            labels = lbs
            
        del fts, lbs

    if split_size > 0:
        x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=split_size,
                                                              random_state=9)
        train_set = data.TensorDataset(torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_train, dtype=torch.float32))
        valid_set = data.TensorDataset(torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_valid, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_set, batch_size)
    else:
        train_set = data.TensorDataset(torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(labels, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = None

    return train_loader, valid_loader

def mn_save_model(model, name): # aim save model
    work_dir =  './work_dir_mn0518'
    model_path = '{}/{}'.format(work_dir, name)
    print(model_path)
    state_dict = model.state_dict()
    weights = OrderedDict([[''.join(k.split('module.')),
                            v.cpu()] for k, v in state_dict.items()])
    torch.save(weights, model_path)
    print('The model has been saved as {}.'.format(model_path))

def accuracy_batch(y_pred, y_true):
    return (y_pred.argmax(1) == y_true.argmax(1)).mean()


def set_training(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode
    model.train(mode)
    return model

def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds-target)**2/(2*variance))
    if add_const:
        const = 0.5*np.log(2*np.pi*variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))

def kl_categorical(preds, log_prior, num_node, eps=1e-16):
    kl_div = preds*(torch.log(preds+eps)-log_prior)
    return kl_div.sum()/(num_node*preds.size(0))

def mn_load_weights( model, weights_path, ignore_weights=None):
    if ignore_weights is None:
        ignore_weights = []
    if isinstance(ignore_weights, str):
        ignore_weights = [ignore_weights]

    print('Load weights from {}.'.format(weights_path))
    weights = torch.load(weights_path)
    weights = OrderedDict([[k.split('module.')[-1],
                            v.cpu()] for k, v in weights.items()])

        # filter weights
    for i in ignore_weights:
        ignore_name = list()
        for w in weights:
            if w.find(i) == 0:
                ignore_name.append(w)
        for n in ignore_name:
            weights.pop(n)
            print('Filter [{}] remove weights [{}].'.format(i,n))

    for w in weights:
        print('Load weights [{}].'.format(w))

    try:
        model.load_state_dict(weights)
    except (KeyError, RuntimeError):
        state = model.state_dict()
        diff = list(set(state.keys()).difference(set(weights.keys())))
        for d in diff:
            print('Can not find weights [{}].'.format(d))
        state.update(weights)
        model.load_state_dict(state)
    return model

if __name__ == '__main__':
    save_folder = os.path.join(os.path.dirname(__file__), save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # DATA.
    print('data---',data_files[0:1])
    train_loader, _ = load_dataset(data_files[0:1], batch_size)
    valid_loader, train_loader_ = load_dataset(data_files[0:1], batch_size, 0.2)

    train_loader = data.DataLoader(data.ConcatDataset([train_loader.dataset, train_loader_.dataset]),
                                   batch_size, shuffle=True)
    dataloader = {'train': train_loader, 'valid': valid_loader}
    del train_loader_

    # MODEL.
    graph_args = {'strategy': 'spatial'}
    model = TwoStreamSpatialTemporalGraph(graph_args, num_class).to(device)
    #model_autoencoder = AdjacencyLearn(90,80,3,3,80,14).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = Adadelta(model.parameters())

    losser = torch.nn.BCELoss()


#####################################################
#######################################################
    model3 = Autolearn(60,128,10,10,128,14).cuda()
    optimizer3 = optim.Adam(params=model3.parameters(),lr=0.005)
    criterion = nn.MSELoss()


    loss_list = {'train': [], 'valid': []}
    accu_list = {'train': [], 'valid': []}
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []
    training_A = False



    model.load_state_dict(torch.load(os.path.join(save_folder, 'tsstg-model.pth')))

    # EVALUATION.
    model = set_training(model, False)
    data_file = data_files[0]
    eval_loader, _ = load_dataset([data_file], 32)

##########################
    encoder_file = os.path.join('./work_dir_mn0518/epoch100_model3.pt')
    model3.eval()
    model3.load_state_dict(torch.load(encoder_file))
        

##########################


    print('Evaluation.')
    run_loss = 0.0
    run_accu = 0.0
    y_preds = []
    y_trues = []
    with tqdm(eval_loader, desc='eval') as iterator:
        for pts, lbs in iterator:
            pts = pts[:, :2, :, :]
            mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
            mot = mot.to(device)
            pts = pts.to(device)
            #print(np.shape(pts))
            #print(pts[0][:][0][:])
            lbs = lbs.to(device)
            out_auto = model3(pts)
            out_auto = out_auto.view(out_auto.size(0), out_auto.size(1), 30,2)
            out_auto = out_auto.permute(0, 3, 2, 1).contiguous()
            out = model((out_auto, mot))

            loss = losser(out, lbs)

            run_loss += loss.item()
            accu = accuracy_batch(out.detach().cpu().numpy(),
                                  lbs.detach().cpu().numpy())
            run_accu += accu
            #print('check out',out)
            #print('check out000',out[0])
            #print('check lbs',lbs)
            #print('check lbs0000',lbs[0])
            y_preds.extend(out.argmax(1).detach().cpu().numpy())
            y_trues.extend(lbs.argmax(1).cpu().numpy())

            iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(
                loss.item(), accu))
            iterator.update()

    run_loss = run_loss / len(iterator)
    run_accu = run_accu / len(iterator)

    cf_matrix = confusion_matrix(y_trues,y_preds)
    print(cf_matrix)
    #plot_confusion_metrix(y_trues, y_preds, class_names, 'Eval on: {}\nLoss: {:.4f}, Accu{:.4f}'.format(
    #    os.path.basename(data_file), run_loss, run_accu
    #), 'true', save=os.path.join(save_folder, '{}-confusion_matrix.png'.format(
    #    os.path.basename(data_file).split('.')[0])))

    print('Eval Loss: {:.4f}, Accu: {:.4f}'.format(run_loss, run_accu))
