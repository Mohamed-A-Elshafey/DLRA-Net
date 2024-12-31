import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import time
import dataset # NITRE2022 Data set
from utils import initialize_logger, save_checkpoint, record_loss, Loss_train, Loss_valid
import visdom
from train import train,test
from DLRA import DLRA




parser = argparse.ArgumentParser(description="SSR")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--end_epoch", type=int, default=130+1, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--decay_power", type=float, default=1.5, help="decay power")
parser.add_argument("--max_iter", type=float, default=3000000, help="max_iter")  
parser.add_argument("--outf", type=str, default="Results", help='path log files')
parser.add_argument('--b1', type = float, default = 0.9, help = 'Adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
opt = parser.parse_args()


def start():
    cudnn.benchmark = True

    # load dataset 
    print("\nloading dataset ...")
    train_dataset = dataset.HyperDataset(mode='train')
    test_dataset  = dataset.HyperDataset(mode='test')

    print("Train_dataset:%d" % (len(train_dataset)))
    print("Validation set samples:", len(test_dataset))
    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(dataset=test_dataset, batch_size=1,  shuffle=False, num_workers=0, pin_memory=True)
    

    
    #torch.autograd.set_detect_anomaly(True)
    # visualzation
    viz = visdom.Visdom(env="DLRA-Net")
    if not viz.check_connection():
        print("Visdom is not connected. Did you run 'python -m visdom.server' ?")
    # model
    print("\nbuilding models_baseline ...")
 
    model = DLRA(3,31,80,10)
    print(f" Torch version: {torch.__version__}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f" GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU properties: {torch.cuda.get_device_properties(0)}")
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    criterion_train = Loss_train()
  

    criterion_valid = Loss_valid()
    if torch.cuda.device_count() > 1:
        print("Do Parallel")
        model = nn.DataParallel(model)  # batchsize integer times
    if torch.cuda.is_available():
        model.cuda()
        criterion_train.cuda()
        criterion_valid.cuda()                                     

    # Parameters, Loss and Optimizer
    start_epoch = 0
    iteration = 0
    record_val_loss = 1000
    optimizer = optim.Adam(model.parameters(), lr = opt.init_lr, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    # Record
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    loss_csv = open(os.path.join(opt.outf, 'loss.csv'), 'a+')
    log_dir = 'Results/train.log'
    logger = initialize_logger(log_dir)

    # Resume
    resume_file = opt.outf + '/trained.pth'
    #resume_file = ''    if resume_file:
    if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file,weights_only=True)
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    # start epoch
    for epoch in range(start_epoch, opt.end_epoch):
        start_time = time.time()
        print("start Training...")
        train_loss,iteration = train(train_loader, model, criterion_train, optimizer, epoch, opt)
        print("start testing...")
        val_loss = test(model, val_loader, criterion_valid)
        # Save model either the best model so far or every 10 epochs 
        if torch.abs(val_loss - record_val_loss) < 0.0001 or val_loss < record_val_loss:
             print("Saving model")
             save_checkpoint("Results/", epoch, iteration, model, optimizer,best=True)
             if val_loss < record_val_loss:
                 record_val_loss = val_loss
        if epoch %5==0:
               save_checkpoint( "Results/", epoch, iteration, model, optimizer,best=False)
        # print loss
        end_time = time.time()
        epoch_time = end_time - start_time
        print("Epoch [%02d], Iter[%06d], Time:%.9f, Train Loss: %.9f Test Loss: %.9f "
               % (epoch, iteration, epoch_time, train_loss, val_loss))
        
        
        #for learning rate
        viz.line([optimizer.param_groups[0]['lr']*(10**4)],[epoch],win='Learning rate schedule',update='append',
          opts=dict(title='Learning rate schedule',
                          legend=['lr*(10^4)'])) 
        #for HSI train loss
        viz.line([train_loss.detach().cpu()],[epoch],win='HSI Train_loss', 
                  update='append',opts=dict(title=' HSI Train Learning Curve.',
                                            legend=['HSI Train Loss']))
        #for validation loss
        viz.line([val_loss.detach().cpu()],[epoch],win='Val Train_loss', 
                  update='append',opts=dict(title='val loss Learning Curve.',
                                            legend=['val loss']))
        # for train_loss and validation_loss
        viz.line([[train_loss.detach().cpu(),val_loss.detach().cpu()]],[epoch],win='Train_loss and val_loss', 
                  update='append',opts=dict(title='Learning Curve.',
                                            legend=['Train Loss', 'Validation Loss']))
        
        # save loss
        record_loss(loss_csv,epoch, train_loss, val_loss)
        logger.info("Epoch [%02d], Train Loss: %.9f Validation Loss: %.9f " 
                    % (epoch, train_loss, val_loss))
       


if __name__ == '__main__':
    start()
    print(torch.__version__)
