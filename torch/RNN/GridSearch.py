import torch
from torchvision import datasets, transforms
import os
import json
import numpy
import sys
from AutoEncoderCNN import AE_CNN

class GridSearch:
    '''
    GridSearch object has twelve attributes. six of the attributes are hyperparameters that are meant to be optimized.
    
    '_verbose': 0 for no info, 1 for some info, 2 for most info
    '_test_set': The test set of the images
    '_device': CPU or GPU, training device
    '_best_dict': Dictionary with the best hyperparameters
    '_early_stop_depth': The patience of the early stopping
    '_json_dict': a dictionary with all the info of each combination
    '''
    
    def __init__(self, device,
                 early_stop_depth = 0,
                 epochs=[3], 
                 learning_rate=[0.01],
                 weight_decay=[1e-5],
                 batch_size=[16],
                 first_dim = [64],
                 encode_dim=[128],
                 verbose = 2):
        
        self._verbose = verbose
        if verbose > 2 or verbose < 0:
            self._verbose = 2
        
        torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device
        
        self._test_set = None
        self._val_set = None
        
        # Hyperparams:
        self._epochs = epochs
        self._lr = learning_rate
        self._wd = weight_decay
        self._batch_size = batch_size
        self._first_dim = first_dim
        self._encode_dim = encode_dim
        
        #Best train dict
        self._best_dict = {"Epochs": 0, "Learning Rate": 0, 
                           "Weight Decay" : 0, "Batch Size": 0, 
                           "First Dim": 0, "Encode Dim": 0,
                           "Model" : False, "Loss": 0}
        self._early_stop_depth = early_stop_depth
        
        self._json_dict = {}
                
    def get_loader(self, BATCH_SIZE):
        '''
        Creates a loader object and test set
        
        Parameters: 'BATCH_SIZE': The batch_size for the loader
        
        Returns: tuple with a loader object and the test set
        '''
        
        # change to subImages_slide299
        #PATH = '/groups/francescavitali/eb2/subImages2/H&E' # has 204 images
        PATH = '/groups/francescavitali/eb2/subImages_slide299/H&E' # has 506 images

        tensor_transform = transforms.ToTensor()

        dataset = datasets.ImageFolder(PATH, 
                                      transform = tensor_transform) #loads the images

        train_set, val_set, test_set = torch.utils.data.random_split(dataset,
                                                           [404,51,51],# 70%, 30%
                                                           generator=torch.Generator(device=self._device))

        loader = torch.utils.data.DataLoader(dataset = train_set,
                                            batch_size = BATCH_SIZE,
                                            shuffle = True,
                                            generator=torch.Generator(device=self._device))
        return (loader, val_set, test_set)
        
        
    def search(self):
        '''
        Runs the full grid search with all the parameters. Saves each info in ''_json_dict' and updates the '_best_dict' accordingly.
        
        Parameters: None
        
        Returns: None
        '''
        
        count = 0
        print(f"Starting search with {len(self._epochs) * len(self._lr) * len(self._wd) * len(self._batch_size) * len(self._first_dim) * len(self._encode_dim)} combinations and a early stopping patience of: {self._early_stop_depth}\n") 
              
        for bs in self._batch_size:
            loader, val_set, test_set = self.get_loader(bs)
            
            if not self._test_set:
                self._test_set = test_set
                self._val_set = val_set
            
            for first_dim in self._first_dim:
                for encode_dim in self._encode_dim:
                    for learning_rate in self._lr:
                        for weight_decay in self._wd:
                            for amt_epochs in self._epochs:
                                if self._verbose != 0:
                                    print(f'---Count: {count}, Epochs: {amt_epochs}, Weight_Decay: {weight_decay}, Learning_Rate: {learning_rate}, Batch_Size: {bs}, First Dim: {first_dim}, Encode Dim: {encode_dim}---\n')
                                else:
                                    print("Next cycle")

                                trained_model, loss = self.training(loader, learning_rate, weight_decay, amt_epochs, first_dim, encode_dim)
                                cur_dict = {"Epochs": 0, "Learning Rate": 0, 
                                           "Weight Decay" : 0, "Batch Size": 0, 
                                           "First Dim": 0, "Encode Dim": 0,
                                           "Model" : False, "Loss": 0}
                                
                                cur_dict["Epochs"] = amt_epochs
                                cur_dict["Learning Rate"] = learning_rate
                                cur_dict["Weight Decay"] = weight_decay
                                cur_dict["Batch Size"] = bs
                                cur_dict["First Dim"] = first_dim
                                cur_dict["Encode Dim"] = encode_dim
                                cur_dict["Model"] = True
                                cur_dict["Loss"] = loss
                                
                                if not self._best_dict["Model"] or self._best_dict["Loss"] > loss:
                                    self._best_dict = cur_dict
                                    torch.save(trained_model.state_dict(), f'./models/model_gs.pth')
                                    if self._verbose != 0:
                                        print(f"Updated Dict \n{self._best_dict}\n")

                                count += 1
                                self._json_dict[count] = cur_dict
                                
        print(f"Best Dict: \n{self._best_dict}")

                        
                   
        
        
    def training(self, loader, lr, weight_decay, EPOCHS, first_dim, encode_dim):
        '''
         Trains the changed model with the new hyperparameters.
        
         Parameters: 'loader': the image loader
                     'lr': learning rate
                     'weight_decay': weight decay
                     'EPOCHS': the amount of epochs to run for
                     'first_dim': the dimensions of the first compression
                     'encode_dim': the dimensions of the encoded compression

         Returns: The trained model and the average? loss
        '''
            
        model = AE_CNN(first_dim, encode_dim).to(self._device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
        #loss_function = torch.nn.BCELoss()
        loss_function = torch.nn.MSELoss()

        
        loss_arr = []
        min_loss = None
        outputs = []
        early_stop = False
        early_stop_depth = self._early_stop_depth
        
        for epoch in range(EPOCHS):
            
            if early_stop:
                if self._verbose != 0:
                    print(f'\n\n------EARLY STOP {min_loss}------\n\n')
                break

            count = 0

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
                if self._verbose == 2:
                    sys.stdout.write('\r')
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
            
            if self._verbose != 0:
                print(f'\nEpoch: {epoch + 1} | Loss: {loss.item():.4f}', end='\n'*2)


        return (model, loss.item())
    
    def _save_json(self, save_dict, path):
        '''
         Saves a dictionary to the specified path
        
         Parameters: 'save_dict': the dictionary to save
                     'path': the path in which to save
        
         Returns: None
        '''
            
        if os.path.exists(path):
            os.remove(path)
        with open(path, "w") as outfile: 
            json.dump(save_dict, outfile, indent=4)
    
    def save_dicts(self):
        '''
         Saves both attribute dictionaries to their respective files.
        
         Parameters: None
        
         Returns: None
        '''
        PATH = "./json/"
        self._save_json(self._best_dict, PATH + "best_dict.json")
        print(f"Saved best dict: {self._best_dict}\n\n")
        
        self._save_json(self._json_dict, PATH + "all_dict.json")
        print(f"Saved all info: {self._json_dict}\n\n")
        
   
