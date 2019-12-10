import torch
tensor = torch.ones((2,), dtype=torch.float64)
t1 = tensor.new_full((32,512,1,1), 3.14)
print(tensor.shape)
t2 = tensor.new_full((32,512,1,1), 23.54)
t3 = tensor.new_full((32,512,1,1), 3.6758679)
t4 = tensor.new_full((32,512,1,1), 1.3453)

bigboi = torch.cat((t1,t2,t3,t4), 0)

print(bigboi.shape)

torch.save(bigboi, 'testing.pt')

class TensorDataset(torch.utils.data.TensorDataset):
    def __init__(self, tensors):
        super(TensorDataset).__init__()
        self.tensors = tensors
    
    def __len__(self):
        return self.tensors.shape[0]
    
    def __getitem__(self, idx):
        return self.tensors[idx]

dataset = TensorDataset(torch.load('testing.pt'))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

for data in dataloader:
    print(data.shape)