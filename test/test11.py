from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
print()
print(dataset)
print(dataset[0]["text"])  # 最初のレビューのテキスト内容を表示

pipe = pipeline("text-classification", device=0)
for out in pipe(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
