from soft_dtw_cuda import SoftDTW
import torch
import numpy as np
import datetime
from Screener import Screener as screener
from Test import Data as data
from Test import Main
from tqdm import tqdm
import asyncio
from torch.utils.data import DataLoader, TensorDataset

async def process_batch(batch, sdtw):
    x_batch, y, tickers, indices = batch
    x_tensor = torch.from_numpy(x_batch).float().cuda()

    y_tensor = torch.from_numpy(y).float().unsqueeze(0).cuda()
    print(f'{x_tensor.size} , {y_tensor.size}')
    loss = sdtw(x_tensor, y_tensor)
    
    return [(ticker, loss.mean().item(), index) for ticker, index in zip(tickers, indices)]
def gpu_dtw(bar):
    x, y, sdtw,ticker,index = bar
    x_tensor = torch.from_numpy(x).float().unsqueeze(0).cuda()
    y_tensor = torch.from_numpy(y).float().unsqueeze(0).cuda()

    loss = sdtw(x_tensor, y_tensor)
    return [ticker,loss.mean().item(),index]



def cuda_convert(x, y):
    x = x.cuda()
    y = y.cuda()
    return x, y


class Match:
    def fetch(ticker,bars=10,dt = None):
        
        tf = 'd'
        if dt != None:
            df = data(ticker,tf,dt,bars = bars+1)
        else:
            df = data(ticker,tf)
        df.np(bars,True)

        return df


    



    # Create a DataLoader for your data
async def main(dataloader,sdtw):
    scores = []
    dataset = TensorDataset(torch.tensor(x_list), torch.tensor(y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    pbar = tqdm(total=len(dataloader.dataset))
    for batch in dataloader:
        x_batch, y, tickers, indices = batch
        result = await asyncio.gather(process_batch((x_batch, y, tickers, indices), sdtw))
        scores.extend(result[0])
        pbar.update(x_batch.shape[0])



if __name__ == '__main__':
    ticker_list = screener.get('full')[:50]
    dfs = Main.pool(Match.fetch,ticker_list)
    x_list = []
    for df in dfs:
        ticker = df.ticker
        for x,index in df.np:
            #print(x)
            #print(index)
            #x_list.append([x,ticker,index])
            x_list.append(x)
            
    #dfs = Match.match(ticker,dt,bars,dfs)#
    sdtw = SoftDTW(use_cuda=True, gamma=0.1)

    ticker = 'JBL' #input('input ticker: ')
    dt = '2023-10-03' #input('input date: ')
    bars = 10 #int(input('input bars: '))
    start = datetime.datetime.now()
    y = Match.fetch(ticker,bars,dt).np
    y = y[0][0]
    batch_size = 16
    num_workers = 4

    dataset = TensorDataset(torch.tensor(x_list), torch.tensor(y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    asyncio.run(main(dataloader,sdtw))
   # sdtw = SoftDTW(use_cuda=True, gamma=0.1)
    
    #for x,ticker,index in x_list: 

    #    scores.append(gpu_dtw([x, y, sdtw,ticker,index]))
    #    pbar.update(1)
    #pbar.close()

   # scores = []
   # for df in dfs:
        #lis = df.get_scores()
        #scores += lis
    #scores.sort(key=lambda x: x[1])
    print(f'completed in {datetime.datetime.now() - start}')
    for ticker,score,index in scores[:20]:
        print(f'{ticker} {data(ticker).df.index[index]}')