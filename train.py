import warnings
warnings.filterwarnings("ignore")

from dataset import MatchesDataset
from networks import MLP
from torchvision import transforms
import torch.optim
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import time


print_interval = 1000

def main():
    
    criterion = nn.CrossEntropyLoss()

    train_dataset = MatchesDataset()
    test_dataset = MatchesDataset(False)
    model = MLP(num_features=274, num_classes=2).float()
    optimizer = torch.optim.SGD(model.parameters(), 0.0005,
                                momentum=0.9,
                                weight_decay=0.00005)

    train_loader = torch.utils.data.DataLoader(
       train_dataset, batch_size=256, shuffle = True,
       num_workers=2, pin_memory=True)
    

    test_loader = torch.utils.data.DataLoader(
       test_dataset, batch_size=256, shuffle = True,
       num_workers=2, pin_memory=True)
     

    for epoch in range(0,100):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        validate(test_loader, model, criterion)
        state = {
            'epoch': epoch + 1,
            'arch': "MLP",
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }



def train(train_loader, model, criterion, optimizer, epoch):

    
    correct = 0
    total = 0
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        stime = time.time()
        # measure data loading time
        target = target.float()
        input = input.float()
        # compute output
        output = model(input)
        _, predicted = torch.max(output.data,1)
        correct += (predicted.float() == target.float()).sum().item()
        total += target.size(0)
        loss = criterion(output, target.long())
        # measure accuracy and record loss
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        # if i % print_interval == 0:
        #     state = {
        #         'iteration': i+1,
        #         'arch': "MLP",
        #         'state_dict': model.state_dict(),
        #         'optimizer' : optimizer.state_dict(),
        #     }
    TT = time.time() -stime
    print('Epoch: [{0}]\t'
          'Training Loss {loss:.4f}\t'
          'Training Acc {acc:.4f}\t'
          'Time: {time:.2f}\t'.format(
           epoch,loss=loss.data[0], acc =(100*correct/total), time= TT))




def validate(val_loader, model, criterion):

    # switch to evaluate mode
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():

        for i, (input, target) in enumerate(val_loader):

            target = target.float()
            # compute output
            input = input.float()
            output = model(input)
            _, predicted = torch.max(output.data,1)
            correct += (predicted.float() == target.float()).sum().item()
            total += target.size(0)
            loss = criterion(output, target.long())

    print('Test: \t'
          'Loss {loss:.4f}\t'
          'Testing Acc {acc:.4f}\t'.format(
            acc =(100*correct/total), loss=loss))
           
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'model_best.pth.tar')

    
main()