import torch
from uni import get_encoder

if __name__ == "__main__":
    encoder, transform = get_encoder(enc_name='uni2-h')
    torch.save(encoder, f'models/encoder.pth')
    torch.save(transform, f'models/transform.pth')
    print(encoder)