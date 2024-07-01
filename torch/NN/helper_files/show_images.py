import numpy as np
import matplotlib.pyplot as plt
import torch

def show_images(img_set, model, amt=10):
    encoder = model.encoder
    decoder = model.decoder
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    auto_encoder_output = []
    encoded_imgs_arr = []
    decoded_imgs_arr = []

    for x in range(len(img_set)):
        with torch.no_grad():
            inp = img_set.__getitem__(x)[0].to(device).flatten()

            encoded_imgs = encoder(inp)
            encoded_imgs_arr.append(encoded_imgs)

            decoded_imgs = decoder(encoded_imgs)
            decoded_imgs_arr.append((inp, decoded_imgs))

            auto_encoder_output.append((inp, model(inp)))
            
    
    
    
    plt.figure(figsize=(20,4))
    for k in range(amt):
        ax = plt.subplot(4, amt, k+1)

        img = decoded_imgs_arr[k][0]
        img = torch.unflatten(img, 0, (3, 299, 299)).detach().cpu().numpy()
        plt.imshow(np.transpose(img, (1,2,0))) # changes to rgb
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4,amt, k+1+amt)
        plt.gray()

        img = encoded_imgs_arr[k].detach().cpu().numpy()
        img = img.reshape(-1,32,16)
        plt.imshow(img[0])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


        ax = plt.subplot(4,amt, k+1+2*amt)
        recon = decoded_imgs_arr[k][1]
        recon = torch.unflatten(recon, 0, (3, 299, 299)).detach().cpu().numpy()
        plt.imshow(np.transpose(recon, (1,2,0)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4,amt, k+1+3*amt)

        recon = auto_encoder_output[k][1]
        recon = torch.unflatten(recon, 0, (3,299,299)).detach().cpu().numpy()
        plt.imshow(np.transpose(recon, (1,2,0)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)