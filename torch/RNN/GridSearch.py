import torch
from torchvision import datasets, transforms
import os
import numpy
import sys
from AutoEncoderCNN import AE_CNN

class GridSearch:
    
    def __init__(self, model, device,
                 early_stop_depth = 0,
                 epochs=[3], 
                 learning_rate=[0.01],
                 weight_decay=[1e-5],
                 batch_size=[16]):
        
        torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device
        
        self._test_set = []
        
        # Hyperparams:
        self._epochs = epochs
        self._lr = learning_rate
        self._wd = weight_decay
        self._batch_size = batch_size
        
        #Best train dict
        self._best_dict = {"Epochs": 0, "Learning Rate": 0, 
                           "Weight Decay" : 0, "Batch Size": 0, 
                           "Model" : False, "Loss": 0}
        self._early_stop_depth = early_stop_depth
                
    def get_loader(self, BATCH_SIZE):
        PATH = '/groups/francescavitali/eb2/subImages2/H&E'

        tensor_transform = transforms.ToTensor()

        dataset = datasets.ImageFolder(PATH, 
                                      transform = tensor_transform) #loads the images

        train_set, test_set = torch.utils.data.random_split(dataset,
                                                           [194,10],# 70%, 30%
                                                           generator=torch.Generator(device=self._device))

        loader = torch.utils.data.DataLoader(dataset = train_set,
                                            batch_size = BATCH_SIZE,
                                            shuffle = True,
                                            generator=torch.Generator(device=self._device))
        return (loader, test_set)
        
        
    def search(self):
        count = 0
        for bs in self._batch_size:
            loader, test_set = self.get_loader(bs)
            
                  
            for learning_rate in self._lr:
                for weight_decay in self._wd:
                    for amt_epochs in self._epochs:
                        print(f'Count: {count}, Epochs: {amt_epochs}, Weight_Decay: {weight_decay}, Learning_Rate: {learning_rate},Batch_Size: {bs}')

                        trained_model, loss = self.training(loader, learning_rate, weight_decay, amt_epochs)
                        if not self._best_dict["Model"] or self._best_dict["Loss"] > loss:
                            self._best_dict["Epochs"] = amt_epochs
                            self._best_dict["Learning Rate"] = learning_rate
                            self._best_dict["Weight Decay"] = weight_decay
                            self._best_dict["Batch Size"] = bs
                            self._best_dict["Model"] = True
                            self._best_dict["Loss"] = loss
                            torch.save(trained_model.state_dict(), f'./models/model_gs.pth')
                            print(f"Updated Dict \n{self._best_dict}")

                        count += 1
                        self._test_set.append(test_set)
        print(f"Best Dict \n{self._best_dict}")

                        
                   
        
        
    def training(self, loader, lr, weight_decay, EPOCHS):
        
        model = AE_CNN().to(self._device)
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
        loss_function = torch.nn.BCELoss()
        
        loss_arr = []
        min_loss = None
        outputs = []
        early_stop = False
        early_stop_depth = self._early_stop_depth
        
        for epoch in range(EPOCHS):
            
            if early_stop:
                print(f'\n\n------EARLY STOP {min_loss}------\n\n')
                break

            count = 0
            loss_total = 0

            model.train()
            for (image, _) in loader:
                image = image.to(self._device)
                #image = image.flatten(start_dim=1) # ignore the batch_size

                recon = model(image)
                loss = loss_function(recon, image)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # UI
                sys.stdout.write('\r')
                loss_total += loss.item()
                sys.stdout.write("Epoch: {} [{:{}}] {:.1f}% | Loss: {}".format(epoch+1, "="*count, 
                                                                               len(loader)-1, 
                                                                               (100/(len(loader)-1)*count), 
                                                                               loss.item()))
                sys.stdout.flush()

                count += 1
                
            loss_arr.append(loss.item())
            if not min_loss:
                min_loss = loss_arr[0]
            if early_stop_depth >= 1 and early_stop_depth < len(loss_arr[loss_arr.index(min_loss):]):
                early_stop = True
                for loss_item in loss_arr[loss_arr.index(min_loss):]:
                    if loss_item < min_loss:
                        min_loss = loss_item
                        early_stop = False
               
                    
            outputs.append((epoch, image[1], recon[1]))

            print(f'\nEpoch: {epoch + 1} | Loss: {(loss_total)/len(loader):.4f}', end='\n'*2)


        return model, (loss_total/len(loader))
