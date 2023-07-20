from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
class train_test_evaluate:
  def __init__(self):
  ############################## Training  ###########################################
    self.train_losses = []
    self.test_losses = []
    self.train_acc = []
    self.test_acc = []

  def train(self,model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    train_loss = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(pbar):
      # get samples
      data, target = data.to(device), target.to(device)

      # Init
      optimizer.zero_grad()
      # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
      # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.
      # Predict
      y_pred = model(data)

      # Calculate loss
      #loss = F.nll_loss(y_pred, target)
      loss = criterion(y_pred, target)
      #train_losses.append(loss)
      train_loss +=loss.item()
      # Backpropagation
      loss.backward()
      optimizer.step()
      # Update pbar-tqdm
      pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      processed += len(data)
      pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    self.train_losses.append(loss.item()) # train_loss
    self.train_acc.append(100*correct/processed)
    #return train_losses, train_acc

  def test(self,model, device, test_loader):
      model.eval()
      test_loss = 0
      correct = 0
      criterion = nn.CrossEntropyLoss()
      with torch.no_grad():
          for data, target in test_loader:
              data, target = data.to(device), target.to(device)
              output = model(data)
              #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
              test_loss += criterion(output, target).item()
              pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              correct += pred.eq(target.view_as(pred)).sum().item()

      test_loss /= len(test_loader.dataset)
      self.test_losses.append(test_loss)

      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))

      self.test_acc.append(100. * correct / len(test_loader.dataset))
      return test_loss

  def update_lr(self,optimizer, lr):
      for g in optimizer.param_groups:
          g['lr'] = lr


  def OneCyclePolicy(self,LRmax, step, iterations):
      LRmin = LRmax/10;
      LRt = LRmin
      LRvalues =[]
      for x in range(iterations):
          if (x<=step):
            LRt += (LRmax - LRmin)/step
            LRvalues.append(LRt)
          else:
            LRt -= (LRmax - LRmin)/(iterations-step)
            LRvalues.append(LRt)
      return LRvalues



  def Training(self,epochs,model,device,optimizer, trainloader, testloader, LR):
    Testloss = 0
    #LRvalues = self.OneCyclePolicy(LR, 5, epochs)
    #optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.95)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(trainloader), epochs=epochs)
    for epoch in range(epochs):
        print("EPOCH:", epoch)
        #self.update_lr(optimizer,LRvalues[epoch])
        self.train(model, device, trainloader, optimizer, epoch)
        Testloss = self.test(model, device, testloader)
        scheduler.step()

  def plotPerformanceGraph(self):
    import matplotlib.pyplot as plt
    fig, (axs1,axs2) = plt.subplots(2, 1,figsize=(15,10))

    axs1.plot(self.train_losses, label = " Train Loss")
    axs1.plot(self.test_losses, label = " Test Loss")
    axs1.set_title(" Loss")

    axs2.plot(self.train_acc, label = " Train Accuracy")
    axs2.plot(self.test_acc, label = " Test Accuracy")

    axs2.set_title(" Accuracy")
    axs1.legend()
    axs2.legend()
    plt.show()

  def MissClassifedImage(self,dataSet, model,device, dispCount,classes):
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    dataiter = iter(dataSet)
    fig, axs = plt.subplots(int(math.ceil(dispCount/5)),5,figsize=(10,10))
    fig.tight_layout()
    count =0
    while True:
        if count >= dispCount:
          break
        images, labels = next(dataiter)
        imagex = images
        images, labels = images.to(device), labels.to(device)
        model= model.to(device)
        output = model(images)
        a, predicted = torch.max(output, 1)
        if(labels != predicted):
          imagex = imagex.squeeze()
          imagex = np.transpose(imagex, (1, 2, 0))
          axs[int(count/5), count%5].imshow(imagex)
          axs[int(count/5), count%5].set_title("Orig: "+str(classes[labels])+", Pred: "+str(classes[predicted]))
          fig.tight_layout(pad=3.0)
          count = count +1
    plt.show()

  def ClassTestAccuracy(self,testloader,device,model, classes):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images =images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
