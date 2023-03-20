# %% [code]
# %% [code]
from alphajanggi_utils import *
import torch
import torch.multiprocessing as mp
from numpy.random import shuffle

def OneProcess(model_alpha, iter_num, lr, n=32, think_time=1024, train_num=3, sample_num=4096, batch_size=256):
    optimizer = torch.optim.Adam(model_alpha.parameters(), lr = lr, weight_decay=1e-6)
    for i in range(iter_num):
        model_alpha.eval()
        samples = SelfPlay_Multi(model_alpha, n = n, think_time=think_time)
        print(len(samples))
        model_alpha.train()
        for _ in range(train_num):
            Train(model_alpha, optimizer, samples, sample_num = sample_num, batch_size=batch_size)
        torch.save(model_alpha.state_dict(), 'AlphaJanggi.pth')

def TwoProcess(model_alpha, model_alpha2, iter_num, lr, n=16, think_time=1024, train_num=2, sample_num=4096, batch_size=256):
    model_alpha2.eval()
    optimizer = torch.optim.AdamW(model_alpha.parameters(), lr = lr, weight_decay=1e-5)
    #total_samples = LoadSamples('/kaggle/input/alphajanggi-dataset/AlphaJanggi_samples')
    total_samples = LoadSamples('/kaggle/input/alphajanggi-samples/AlphaJanggi_samples')
    steps = 8192
    with mp.Pool(2) as p:
        for i in range(iter_num):
            model_alpha.eval()
            model_alpha2.load_state_dict(model_alpha.state_dict())
            samples, samples2 = p.starmap(Alpha_SelfPlay_Multi, [(model_alpha, n, think_time), (model_alpha2, n, think_time)])
            torch.cuda.empty_cache()
            samples+=samples2
            print(len(samples))
            total_samples += samples
            if len(total_samples)>40000:
                total_samples = total_samples[-40000:]
            print(len(total_samples))
            model_alpha.train()
            for _ in range(train_num):
                steps += Train(model_alpha, optimizer, total_samples, sample_num = sample_num, batch_size=batch_size)
            torch.cuda.empty_cache()
            print(steps, "steps")
    torch.save(model_alpha.state_dict(), 'AlphaJanggi.pth')
    SaveSamples(total_samples, 'AlphaJanggi_samples')
    
    
def TwoProcess_notrain(model_alpha, model_alpha2, iter_num, n=16, think_time=1024):
    model_alpha.eval()
    model_alpha2.eval()
    #total_samples = LoadSamples('/kaggle/input/alphajanggi-dataset/AlphaJanggi_samples')
    #total_samples = LoadSamples('/kaggle/input/alphajanggi-samples/AlphaJanggi_samples')
    with mp.Pool(2) as p:
        for i in range(iter_num):
            model_alpha2.load_state_dict(model_alpha.state_dict())
            samples, samples2 = p.starmap(Alpha_SelfPlay_Multi, [(model_alpha, n, think_time), (model_alpha2, n, think_time)])
            torch.cuda.empty_cache()
            samples+=samples2
            print(len(samples))
            #total_samples += samples
            total_samples = samples
            if len(total_samples)>40000:
                total_samples = total_samples[-40000:]
            print(len(total_samples))
            torch.cuda.empty_cache()
    torch.save(model_alpha.state_dict(), 'AlphaJanggi.pth')
    SaveSamples(total_samples, 'AlphaJanggi_samples2')

def main():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count()>=2:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
        print("Two cuda")
        model_alpha = AlphaJanggi().to(device0)
        model_alpha.device=device0
        #model_alpha.load_state_dict(torch.load('/kaggle/input/alphajanggi-dataset/AlphaJanggi.pth', map_location = device0))
        model_alpha.load_state_dict(torch.load('/kaggle/input/alphajanggi-samples/AlphaJanggi.pth', map_location = device0))
        model_alpha2 = AlphaJanggi().to(device1)
        model_alpha2.device = device1
        #TwoProcess(model_alpha, model_alpha2, 3, 3e-5, 32, 2048, 1, 32768, 256)
        TwoProcess_notrain(model_alpha, model_alpha2, 1, 32, 4096)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        device = torch.device(device)
        model_alpha = AlphaJanggi().to(device)
        model_alpha.device=device
        OneProcess(model_alpha, 3, 1e-2)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()