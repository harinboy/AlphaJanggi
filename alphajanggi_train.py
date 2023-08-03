from AlphaJanggi_utils import *
import torch
import torch.multiprocessing as mp
from numpy.random import shuffle

def main():
    print(torch.cuda.device_count())
    torch.backends.cudnn.benchmark = True
    device0 = torch.device("cuda:0")
    #print("Two cuda")
    model_alpha = AlphaJanggi().to(device0)
    model_alpha.device=device0
    #model_alpha.load_state_dict(torch.load('./AlphaJanggi.pth', map_location = device0))
    model_alpha.load_state_dict(torch.load('./AlphaJanggi.pth', map_location = device0))
    total_samples = LoadSamples('./AlphaJanggi_samples4')
    lr=1e-6
    optimizer = torch.optim.AdamW(model_alpha.parameters(), lr = lr, weight_decay=1e-5)
    with open('steps', 'r') as fp:
        steps = int(fp.read())
    #steps = 8704
    train_num = 1
    sample_num = 32768
    batch_size = 256
    for _ in range(train_num):
        steps += Train(model_alpha, optimizer, total_samples, sample_num = sample_num, batch_size=batch_size)
    print(steps, "steps")
    torch.save(model_alpha.state_dict(), './AlphaJanggi.pth')
    with open('steps', 'w') as fp:
        fp.write(str(steps))

    #TwoProcess(model_alpha, model_alpha2, 3, 3e-5, 32, 2048, 1, 32768, 256)
    #MultiProcess_notrain(model_alpha, 1, 4, 16, 4096)
    #MultiProcess(model_alpha, 1, 3e-5, 4, 16, 4096, 2, 16384, 256)
    #OneProcess(model_alpha, 1, 3e-5, 64, 4096, 2, 16384, 256)

if __name__ == '__main__':
    main()
