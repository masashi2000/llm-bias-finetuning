from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm

pipe = pipeline("text-classification", "meta-llama/Llama-3.1-8B-Instruct", device=0)


class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        return "This is a test"


dataset = MyDataset()
print()
print(dataset)
print()

for batch_size in [8, 64, 256]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
        pass
