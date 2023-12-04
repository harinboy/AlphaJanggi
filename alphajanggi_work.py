from AlphaJanggi_utils import *
import torch
import torch.multiprocessing as mp
from numpy.random import shuffle

def OneProcess(model_alpha, iter_num, lr, n=32, think_time=1024, train_num=3, sample_num=4096, batch_size=256):
    optimizer = torch.optim.AdamW(model_alpha.parameters(), lr = lr, weight_decay=1e-5)
    total_samples = LoadSamples('AlphaJanggi_samples2')
    steps = 8192

    for i in range(iter_num):
        model_alpha.eval()
        samples = Alpha_SelfPlay_Multi(model_alpha, n = n, think_time=think_time)
        print(len(samples))
        total_samples+=samples
        print(len(total_samples))
        model_alpha.train()
        torch.cuda.empty_cache()
        for _ in range(train_num):
            steps+=Train(model_alpha, optimizer, total_samples, sample_num = sample_num, batch_size=batch_size)
        print(steps, "steps")
    torch.save(model_alpha.state_dict(), 'AlphaJanggi.pth')
    SaveSamples(total_samples, 'AlphaJanggi_samples')

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
    total_samples = LoadSamples('./AlphaJanggi_samples2')
    with mp.Pool(2) as p:
        for i in range(iter_num):
            model_alpha2.load_state_dict(model_alpha.state_dict())
            samples, samples2 = p.starmap(Alpha_SelfPlay_Multi, [(model_alpha, n, think_time), (model_alpha2, n, think_time)])
            torch.cuda.empty_cache()
            samples+=samples2
            print(len(samples))
            #total_samples += samples
            total_samples += samples
            if len(total_samples)>40000:
                total_samples = total_samples[-40000:]
            print(len(total_samples))
            torch.cuda.empty_cache()
    torch.save(model_alpha.state_dict(), 'AlphaJanggi.pth')
    SaveSamples(total_samples, 'AlphaJanggi_samples2')

def MultiProcess_notrain(model_alpha, iter_num, process_num = 8, n=16, think_time=1024):
    model_alpha.eval()
    total_samples = LoadSamples('./AlphaJanggi_samples4')
    #total_samples = []
    with mp.Pool(process_num) as p:
        for i in range(iter_num):
            samples_list = p.starmap(Alpha_SelfPlay_Multi, [(model_alpha, n, think_time)]*process_num)
            torch.cuda.empty_cache()
            samples = []
            for sample in samples_list:
                samples+=sample
            print(len(samples))
            total_samples += samples
            #total_samples = samples
            if len(total_samples)>40000:
                total_samples = total_samples[-40000:]
            print(len(total_samples))
    #torch.save(model_alpha.state_dict(), 'AlphaJanggi.pth')
    SaveSamples(total_samples, 'AlphaJanggi_samples4')

def MultiProcess(model_alpha, iter_num, lr, process_num = 8, n=16, think_time=1024, train_num = 2, sample_num = 32768, batch_size = 256):
    total_samples = LoadSamples('./AlphaJanggi_samples4')
    optimizer = torch.optim.AdamW(model_alpha.parameters(), lr = lr, weight_decay=1e-5)
    with open("steps", 'r') as fp:
        steps = int(fp.read())
    with mp.Pool(process_num) as p:
        for i in range(iter_num):
            model_alpha.eval()
            torch.cuda.empty_cache()
            samples_list = p.starmap(Alpha_SelfPlay_Multi, [(model_alpha, n, think_time)]*process_num)
            samples = []
            for sample in samples_list:
                samples+=sample
            print(len(samples))
            total_samples += samples
            #total_samples = samples
            if len(total_samples)>40000:
                total_samples = total_samples[-40000:]
            print(len(total_samples))
            torch.cuda.empty_cache()
            model_alpha.train()
            for _ in range(train_num):
                steps += Train(model_alpha, optimizer, total_samples, sample_num = sample_num, batch_size=batch_size)
            print(steps, "steps")
            with open("steps", 'w') as fp:
                fp.write(str(steps))
            torch.save(model_alpha.state_dict(), 'AlphaJanggi.pth')
            SaveSamples(total_samples, 'AlphaJanggi_samples4')

def main():
    print(torch.cuda.device_count())
    torch.backends.cudnn.benchmark = True
    device0 = torch.device("cuda:0")
    #print("Two cuda")
    model_alpha = AlphaJanggi().to(device0)
    model_alpha.device=device0
    #model_alpha.load_state_dict(torch.load('./AlphaJanggi_230313_2.pth', map_location = device0))
    model_alpha.load_state_dict(torch.load('./AlphaJanggi.pth', map_location = device0))
    #TwoProcess(model_alpha, model_alpha2, 3, 3e-5, 32, 2048, 1, 32768, 256)
    MultiProcess(model_alpha, 3, 3e-7, 2, 16, 8192, 1, 32768, 256)
    #MultiProcess_notrain(model_alpha, 1, 2, 16, 8192)
    #OneProcess(model_alpha, 1, 3e-5, 64, 4096, 2, 16384, 256)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
