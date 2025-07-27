#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

class MultiHeadAttention(nn.Module):

    def __init__(self,d_in,d_out,n_ctx,n_head):
        super().__init__()
        d_head = d_out // n_head
        self.cache = (d_out,n_head,d_head)
        
        self.W_Q = nn.Linear(d_in,d_out,bias=False)
        self.W_K = nn.Linear(d_in,d_out,bias=False)
        self.W_V = nn.Linear(d_in,d_out,bias=False)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(n_ctx,n_ctx),diagonal=1)
        )
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(d_out,d_out)

    def forward(self,x):

        b,n_tok,d_in = x.shape
        d_out,n_head,d_head = self.cache

        Q = self.W_Q(x)
        K = self.W_Q(x)
        V = self.W_Q(x)

        Q = Q.reshape(b,n_tok,n_head,d_head)
        K = K.reshape(b,n_tok,n_head,d_head)
        V = V.reshape(b,n_tok,n_head,d_head)

        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        scores = Q @ K.transpose(2,3)
        masked = self.mask[:n_tok,:n_tok].bool()
        scores.masked_fill_(masked,-torch.inf)
        W = torch.softmax(scores/K.shape[-1]**0.5,dim=-1)
        W = self.dropout(W)
        ctx = W @ V
        ctx = ctx.transpose(1,2)
        ctx = ctx.reshape(b,n_tok,d_out)
        out = self.out_proj(ctx)

        return out

class Transformer(nn.Module):

    def __init__(self,cfg):
        super().__init__()

        d = cfg['d_emb']
        self.att = MultiHeadAttention(
            d_in = d,
            d_out = d,
            n_ctx = cfg['n_ctx'],
            n_head = cfg['n_head']
        )
        self.ff = FeedForward(d)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = LayerNomr(d)
        self.norm2 = LayerNomr(d)
    
    def forward(self,x):

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        out = x + shortcut

        return out

class LayerNomr(nn.Module):

    def __init__(self,d):
        super().__init__()

        self.g = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self,x):
        
        eps = 1e-5
        mu = torch.mean(x,dim=-1,keepdim=True)
        var = torch.var(x,dim=-1,keepdim=True)
        norm = (x-mu)/(torch.sqrt(var)+eps)
        
        return self.g * norm + self.b

class FeedForward(nn.Module):

    def __init__(self,d):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d,4*d),
            GELU(),
            nn.Linear(4*d,d)
        )
    
    def forward(self,x):

        return self.layers(x)

class GELU(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self,x):

        a = torch.sqrt(torch.tensor(2/torch.pi))
        th = torch.tanh(a*(x+x**3))

        return 0.5*x*(1+th)

class GPT(nn.Module):

    def __init__(self,cfg):
        super().__init__()

        d = cfg['d_emb']
        n = cfg['n_vocab']

        self.tok_emb = nn.Embedding(n,d)
        self.pos_emb = nn.Embedding(n,d)

        self.trf = nn.Sequential(
            *[Transformer(cfg) for _ in range(cfg['n_layers'])]
        )

        self.dropout = nn.Dropout(0.1)
        self.final_norm = LayerNomr(d)
        self.out_heda = nn.Linear(d,n,bias=False)

    def forward(self,ids):

        n_batch,n_seq = ids.shape

        tok = self.tok_emb(ids)
        pos = self.pos_emb(torch.arange(n_seq))
        x = tok+pos
        
        x = self.dropout(x)
        x = self.trf(x)
        x = self.final_norm(x)
        logits = self.out_heda(x)
        
        return logits

def generate(model,ids,n_next,n_ctx):

    for _ in range(n_next):

        ids_cond = ids[:,-n_ctx:]

        with torch.no_grad():
            logits = model(ids_cond)
        logits = logits[:,-1,:]
        prob = torch.softmax(logits,dim=-1)
        ids_next = torch.argmax(prob,dim=-1,keepdim=True)
        ids = torch.cat((ids,ids_next),dim=1)
    
    return ids

class GPTDataset(Dataset):

    def __init__(self,model,ids,n_ctx):

        self.input_ids,self.target_ids = [],[]
    
        for i in range(len(ids)-n_ctx):

            j = i+n_ctx
            input = torch.tensor(ids[i:j])
            target = torch.tensor(ids[i+1:j+1])

            self.input_ids.append(input)
            self.target_ids.append(target)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx],self.target_ids[idx]

def create_data(model,ids,n_ctx):

    dataset = GPTDataset(model,ids,n_ctx)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    return dataloader

def cal_loss(model,x,y):
    logits = model(x)
    loss = nn.functional.cross_entropy(
        logits.flatten(0,1),y.flatten()
    )
    return loss


def trainer(data,model,optimizer,epochs=10):

    for i in range(epochs):
        model.train()
        
        for x,y in data:
            optimizer.zero_grad()
            loss = cal_loss(model,x,y)
            loss.backward()
            optimizer.step()

        model.eval()
        eval_ids = torch.tensor([[0,1,2,3,4]])
        pred = generate(model,eval_ids,10,5)
        print(f'epoch = ({i}) : {pred}')

torch.set_printoptions(precision=2,sci_mode=False)
torch.manual_seed(123)



cfg ={
    'n_vocab':10,
    'd_emb':768,
    'n_ctx':768,
    'n_layers':12,
    'n_head':12
}

ids = torch.tensor([
    0,1,2,3,4,5,6,7,8,9,
    0,1,2,3,4,5,6,7,8,9,
    0,1,2,3,4,5,6,7,8,9,
    0,1,2,3,4,5,6,7,8,9,
])
model = GPT(cfg)
train_data = create_data(model,ids,5)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=4e-4,
    weight_decay=0.1
)
trainer(train_data,model,optimizer,epochs=5)
