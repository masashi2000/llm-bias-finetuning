from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import SystemMessage
import torch

# モデルとトークナイザーの設定
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # すでに取得済みのモデルへのパス
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # GPUで推論
)

# LLM設定
llm = HuggingFacePipeline(model=model, tokenizer=tokenizer)

# エージェントのペルソナ設定
personas = {
    "Republican_1": "You are a strong supporter of the Republican Party.",
    "Republican_2": "You support the Republican Party and value conservative policies.",
    "Democrat_1": "You are a strong supporter of the Democratic Party.",
    "Democrat_2": "You support the Democratic Party and advocate for progressive policies."
}

# 会話の初期化とテーマ設定
topic = "climate change"  # 話し合うトピック。例えば、気候変動について

# 各エージェントの準備
agents = []
for name, persona in personas.items():
    memory = ConversationBufferMemory()  # 各エージェントに会話メモリを持たせる
    prompt_template = PromptTemplate(input_variables=["history"], template=f"{persona} Here is the conversation so far: {{history}}\nNow respond based on your views.")
    
    # エージェントのLLMチェーン
    agent_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory
    )
    
    agents.append((name, agent_chain))

# シミュレーション実行
conversation_history = ""
turns = 5  # 会話のターン数を指定
for turn in range(turns):
    print(f"\nTurn {turn + 1}")
    for name, agent_chain in agents:
        # 各エージェントが会話に応答する
        response = agent_chain.run({"history": conversation_history})
        print(f"{name}: {response}\n")
        
        # 会話履歴を更新
        conversation_history += f"{name}: {response}\n"

