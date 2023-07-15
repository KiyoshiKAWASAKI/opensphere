import os
import os.path as osp
import yaml
import time
import argparse

import torch
import torch.nn as nn


from builder import build_dataloader, build_from_cfg


# Paths to pretrain models. Save features to the same directory
# TODO: sphere_face_vggface2_sfnet20
pretrain_model_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/" \
                     "sphere_face/sphere_face_vggface2_sfnet20/20220501_194024"

# TODO: spherefaceplus_vggface2_sfnet20
# pretrain_model_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/" \
#                      "sphere_face/spherefaceplus_vggface2_sfnet20/20220422_003856"

# TODO: spherefaceR_v2_HFN_vggface2_sfnet20
# pretrain_model_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/" \
#                      "sphere_face/spherefaceR_v2_HFN_vggface2_sfnet20/20220428_232053"



@torch.no_grad()
def get_feats(net, data, flip=True):
    # extract features from the original
    # and horizontally flipped data
    feats = net(data)
    if flip:
        data = torch.flip(data, [3])
        feats += net(data)

    return feats.data.cpu()




@torch.no_grad()
def test_run(net,
             checkpoint,
             dataloaders):

    # load model parameters
    net.load_state_dict(torch.load(checkpoint))

    for n_loader, dataloader in enumerate(dataloaders):
        # get feats from test_loader
        dataset_feats = []
        dataset_indices = []
        for n_batch, (data, indices) in enumerate(dataloader):
            # collect feature and indices
            data = data.cuda()
            indices = indices.tolist()
            feats = get_feats(net, data)



if __name__ == '__main__':
    # TODO: build network from yaml file
    print("Checking PyTorch with CUDA")
    print(torch.cuda.is_available())

    config_path = os.path.join(pretrain_model_dir, 'config.yml')
    print("Loading config yaml: ", config_path)

    with open(config_path, 'r') as f:
        test_config = yaml.load(f, yaml.SafeLoader)

    print("Building network")
    bkb_net = build_from_cfg(
        test_config['model']['backbone']['net'],
        'model.backbone',
    )
    bkb_net = bkb_net.cuda()
    bkb_net.eval()

    # TODO: Load pre-train model parameters

    # TODO: make data loader

    # TODO: run test and save results

