import torch.utils.data
from data_load import LiverDataSet
import time

path="/home/zhangqy/CT/data/LiTS17/Train/"

dataset=LiverDataSet(path)

works=[0,5,10,15,20,25,30,35]

for a in works:

    train_loader=torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True,num_workers=a,pin_memory=False)
    
    start=time.time()
    for ep in range(5):
        
        for i,(img,label) in enumerate(train_loader):
            pass
    
    end=time.time()
    print('Num_workers: ',a,' Total time: ',(end-start)/5 )