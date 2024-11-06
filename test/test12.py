from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

# IMDBデータセットをロード
dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")

# テキスト生成用パイプラインの設定
pipe = pipeline("text-classification",device=0)

# 各テキストに対して生成を行い、バッチで処理
for out in pipe(KeyDataset(dataset, "text"), batch_size=8, max_length=100, truncation=True):
    print(out)
    # [{'generated_text': '生成されたテキスト...'}] のような形式で出力されます

