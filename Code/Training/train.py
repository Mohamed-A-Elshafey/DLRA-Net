import torch
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, Loss_train, Loss_valid
import datetime
import time
import numpy as np

def train(train_loader, model, criterion_train, optimizer, epoch, opt):
    total_loss =  AverageMeter()
    losses = AverageMeter()
    losses_rgb = AverageMeter()
    prev_time = time.time()
    model.train()
    for i,data  in enumerate(train_loader): 
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        model.zero_grad()
        optimizer.zero_grad()
        fake_hyper  = model.forward(images)
        loss  = criterion_train(fake_hyper, labels)
        loss_all = loss 
        loss_all.backward()
        optimizer.step()
       
        # # Determine approximate time left
        iters_done = epoch *len(train_loader) + i
        # Decaying Learning Rate
        lr = poly_lr_scheduler(optimizer, opt.init_lr, iters_done, max_iter=opt.max_iter, power=opt.decay_power)
        iters_left =opt.end_epoch*len(train_loader) - iters_done
        time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
        prev_time = time.time()
        
        
        #  record loss
        losses.update(loss.data)
        total_loss.update(loss_all.data)
        print('[Epoch:%02d],[Batch NO:%d/%d],[iter:%d],[Time_left=%s],[train_losses.avg=%.9f]'
                 % (epoch, i+1, len(train_loader), iters_done, time_left,losses.avg))
    return  losses.avg ,iters_done 
        
        
        

def test(model, test_dataset, criterion):
    
    model.eval()
    losses = AverageMeter()
 
    for i, data in enumerate(test_dataset):
        images,labels = data
      
        images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
           fake_hyper = model.forward(images)
           loss = criterion(fake_hyper, labels)
           losses.update(loss.data)
           
          
    return losses.avg


# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr 

