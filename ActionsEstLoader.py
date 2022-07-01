import os
import torch
import numpy as np
import copy
from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph
from pose_utils import normalize_points_with_size, scale_pose
from Actionsrecognition.auto_learn import *

class TSSTG(object):
    """Two-Stream Spatial Temporal Graph Model Loader.
    Args:
        weight_file: (str) Path to trained weights file.
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    def __init__(self,
                 weight_file='./TSSTG0518/tsstg-model.pth',   ###./Models/TSSTG/tsstg-model.pth'
                 device='cuda'):
        self.graph_args = {'strategy': 'spatial'}
        self.class_names = ['Normal', 'Fall Down']
        self.num_class = len(self.class_names)
        self.device = device

        self.model = TwoStreamSpatialTemporalGraph(self.graph_args, self.num_class).to(self.device)
        self.model.load_state_dict(torch.load(weight_file))
        self.model.eval()

        self.model3 = Autolearn(60,128,10,10,128,14).cuda()
        self.encoder_file = os.path.join('./work_dir_mn0518/epoch100_model3.pt')
        self.model3.load_state_dict(torch.load(self.encoder_file))

    def predict(self, pts, image_size):
        """Predict actions from single person skeleton points and score in time sequence.
        Args:
            pts: (numpy array) points and score in shape `(t, v, c)` where
                t : inputs sequence (time steps).,
                v : number of graph node (body parts).,
                c : channel (x, y, score).,
            image_size: (tuple of int) width, height of image frame.
        Returns:
            (numpy array) Probability of each class actions.
        """
    

        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        
        pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1)
        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(2, 0, 1)[None, :]
        pts = pts[:,:2,:,:]
        

        mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :] # post(pts[:, :2, 1:, :]) - x,y in future frame  , back(pts[:, :2, :-1, :]) = x,y in previous frame 
        


        mot = mot.to(self.device)
        pts = pts.to(self.device)
        out_auto = self.model3(pts)
        out_auto = out_auto.view(out_auto.size(0), out_auto.size(1), 30,2)
        out_auto = out_auto.permute(0, 3, 2, 1).contiguous()

    
        out = self.model((out_auto, mot))

        return out.detach().cpu().numpy()
