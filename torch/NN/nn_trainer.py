import torch
import sys
from util.custom_loss.custom_loss import mse_loss, sparsity_loss
import matplotlib.pyplot as plt

def ae_train(AE: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim, EPOCHS:int=100, loss_weights:list=[1,0,0], patience:int=0, verbose:int=1) -> tuple:
    
    torch.set_default_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    device = torch.get_default_device()
    
    loss_arr = []
    min_loss = None
    
    outputs = []
    
    early_stop = False
    
    AE.to(device)
    AE.train()
    
    for epoch in range(EPOCHS):
        minimum = min(loss_arr[-patience-1:]) if loss_arr else -1 # ignores for the first iteration
        if check_early_stop(get_early_stop(loss_arr, patience), verbose, minimum): return (outputs, loss_arr)
        
        count = 0
        
        for (image, _) in train_loader:
            image = image.to(device)
            decoded = AE(image)

            loss = loss_weights[0]*mse_loss(decoded, image)  + loss_weights[1]*sparsity_loss(AE.encoded_vector)
            if is_nan(loss.item()): return (outputs, loss_arr)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            ui_secondary(verbose, epoch, count, len(train_loader), loss)
            
            count += 1
            
        outputs.append((image, decoded))
        loss_arr.append(loss.item())
        ui_main(verbose, epoch, loss)
    
    print(outputs)
    print(loss_arr)
    return (outputs, loss_arr)
        
def is_nan(num:float) -> bool:
    if num != num:
        text = f'NaN loss: {num}'
        print(text)
        with open('progress.txt', 'a') as file:
            file.write(text)
    return (num!=num)
    
        
        
        
def get_early_stop(loss_arr:list, early_stop_depth:int) -> bool:
    if early_stop_depth < 1 or not loss_arr: return False

    minimum = min(loss_arr[-early_stop_depth-1:]) # smaller list to check
    return len(loss_arr[loss_arr.index(minimum):]) > early_stop_depth
                
                
def check_early_stop(early_stop:bool, verbose:int, min_loss:float) -> bool:
    if early_stop:
        if verbose != 0:
            text = f'\n\n------EARLY STOP {min_loss}------\n\n'
            print(text)
            with open('progress.txt', 'a') as file:
                file.write(f"{text}")
    return early_stop

def ui_secondary(verbose:int, epoch:int, count:int, length:int, loss:torch.Tensor) -> None:
    if verbose == 2:
        sys.stdout.write('\r')
        sys.stdout.write("Epoch: {} [{:{}}] {:.1f}% | Loss: {}".format(epoch+1, "="*count, 
                                                                   len(train_loader)-1, 
                                                                   (100/(len(train_loader)-1)*count), 
                                                                   loss.item()))
        sys.stdout.flush()

def ui_main(verbose:int, epoch:int, loss:torch.Tensor) -> None:
    text = f'\nEpoch: {epoch + 1} | Loss: {loss.item():.4f}'
    print(text, end='\n'*2)
    with open('progress.txt', 'a') as file:
        file.write(f"{text}\n\n")