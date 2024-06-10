from options.train_options import TrainOptions
from data import create_dataset
import torch.optim as optim
from model import *
from torch.optim import lr_scheduler
if __name__ == '__main__':
    device_ids = [1]
    opt = TrainOptions().parse()

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    iters_perepoch = dataset_size // opt.batch_size
    print('The number of training images = %d' % dataset_size)
    single_epoch_iters = (dataset_size // opt.batch_size)
    total_iters = opt.total_epochs * single_epoch_iters 
    cur_iters = opt.resume_iter + opt.resume_epoch * single_epoch_iters
    start_iter = opt.resume_iter

    generator = Uformer()
    generator = torch.nn.DataParallel(generator, device_ids=device_ids)
    generator = generator.cuda(device=device_ids[0])
    generator.train()
    criterionL1 = nn.L1Loss()
    optimizer_G = optim.AdamW(generator.parameters(), lr=opt.lr, betas=(0.9, 0.99), weight_decay=0.02)
    scheduler_G = lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=100, eta_min=1e-6)
    cur_iters = 0
    print('Start training from epoch: {:05d}; iter: {:07d}'.format(opt.resume_epoch, opt.resume_iter))
    for epoch in range(opt.resume_epoch, opt.total_epochs + 1):    
        for i, data in enumerate(dataset, start=start_iter):
            cur_iters += 1
            hr=data['HR'].cuda(device=device_ids[0])
            lr=data['LR'].cuda(device=device_ids[0])
            output = generator(lr)
            loss = criterionL1(hr,output)
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()
            if cur_iters % opt.print_freq == 0:
                print('===============')
                print("Iter:[%d | %d / %d]" %(epoch+1,cur_iters,iters_perepoch))
                print("Loss_Pix:%f"%(loss.item()))
            if cur_iters % opt.save_iter_freq == 0:
                print("saving ckpt")
                torch.save(generator.state_dict(),'./ckpt/demo_model%03d.pt'%cur_iters)
            if cur_iters % opt.save_latest_freq == 0:
                print("saving lastest ckpt")
                torch.save(generator.state_dict(),'./ckpt/lastest_demo_model.pt')
        scheduler_G.step()
   