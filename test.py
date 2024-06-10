import os
from options.test_options import TestOptions
from data import create_dataset
from utils import utils
from PIL import Image
from tqdm import tqdm
import torch
from model import *

if __name__ == '__main__':
    device_ids = [1]
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    if len(opt.save_as_dir):
        save_dir = opt.save_as_dir
    else:
        save_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  
        if opt.load_iter > 0:  # load_iter is 0 by default
            save_dir = '{:s}_iter{:d}'.format(save_dir, opt.load_iter)
    os.makedirs(save_dir, exist_ok=True)

    print('creating result directory', save_dir)

    network = Uformer(img_size=128,embed_dim=32,win_size=8,token_embed='linear',token_mlp='leff')
    network = torch.nn.DataParallel(network, device_ids=device_ids)
    network.eval()
    network.load_state_dict(torch.load(opt.pretrain_model_path,map_location='cuda:7'))
    #170 .97
    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        inp = data['LR']
        with torch.no_grad():
            output_SR = network(inp)
        img_path = data['LR_paths']     # get image paths
        output_sr_img = utils.tensor_to_img(output_SR, normal=True)

        save_path = os.path.join(save_dir, img_path[0].split('/')[-1]) 
        save_img = Image.fromarray(output_sr_img)
        save_img.save(save_path)


       
