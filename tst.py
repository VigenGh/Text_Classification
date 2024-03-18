import math
import torch
import torch.nn as nn
import torchtext
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
torch.manual_seed(0)

class TextIter(torch.utils.data.Dataset):

  def __init__(self, input_data):

      self.text = input_data['Message'].values.tolist()

  def __len__(self):
      return len(self.text)

  def __getitem__(self, idx):
      return self.text[idx]


# Build vocabulary
# def yield_tokens(data_iter):
#     for text in data_iter:
#         yield tokenizer(text)
#
# data_iter = TextIter(df_train)
# vocab = build_vocab_from_iterator(yield_tokens(data_iter), specials=["<pad>", "<unk>"])
# vocab.set_default_index(vocab["<unk>"])
# a = vocab.get_stoi()
# text = 'this is text'
# seq = [vocab[word] for word in tokenizer(text)]
#
# print(tokenizer(text))
# print(seq)

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * (self.d_model ** 0.5)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SingleHeadattention(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.d_model = d_model
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        x = q @ k.transpose(-2, -1) / (self.d_model ** 0.5)
        x = x.softmax(-1)
        return x @ v


class Multihead(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        df = d_model // h
        self.blocks = nn.ModuleList([SingleHeadattention(d_model, df) for _ in range(h)])
        self.lin_agg = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.blocks], axis=-1)
        x = self.lin_agg(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        return self.dropout(self.norm(x1 + x2))


class FeedForward(nn.Module):
    def __init__(self, d_model, dff, drop=0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_model, dff)
        self.layer2 = nn.Linear(dff, d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.layer2(self.drop(self.layer1(x).relu()))


class SingleEncoder(nn.Module):
    def __init__(self, d_model, dff, h):
        super().__init__()
        self.att = Multihead(h, d_model)
        self.ff = FeedForward(d_model, dff)

        self.res_con1 = ResidualConnection(d_model)
        self.res_con2 = ResidualConnection(d_model)

    def forward(self, x):

        att_part = self.att(x)
        x = self.res_con1(x, att_part)
        ff_part = self.ff(x)
        x = self.res_con2(x, ff_part)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, dff, h, encoder_count):
        super().__init__()
        self.blocks = nn.ModuleList([SingleEncoder(d_model, dff, h) for i in range(encoder_count)])
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)

        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, vocal_size, d_model, dff, h, encoder_count):
        super().__init__()
        assert d_model % h == 0

        self.emb = Embeddings(vocal_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, vocal_size)

        self.encoder = Encoder(d_model, dff, h, encoder_count)

        self.last = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.emb(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(1)
        x = self.last(x)
        return x


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, vocab):
        self.vocab = vocab
        self.text = data["Message"].values.tolist()
        self.labels = [label2id[i] for i in data["Category"].values.tolist()]

    def get_sequance(self, idx):
        tokenizer = get_tokenizer('basic_english')
        sequance = [self.vocab[word] for word in tokenizer(self.text[idx])]
        len_seq = len(sequance)
        return sequance, len_seq

    def get_label(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq, seq_len = self.get_sequance(idx)
        label = self.get_label(idx)

        return seq, label, seq_len


def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    max_len = max(lengths)

    for i in range(len(batch)):
        if len(sequences[i]) != max_len:
            for j in range(len(sequences[i]), max_len):
                sequences[i].append(0)

    return torch.tensor(sequences, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def train(model, dataset, epochs, lr, bs, vocab):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((p for p in model.parameters()
                                  if p.requires_grad), lr=lr)
    train_dataset = TextDataset(dataset, vocab)
    train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=bs, collate_fn=collate_fn, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        total_loss_train = 0
        total_acc_train = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch}") as train_set:
            for train_sequence, train_label in train_set:
                # Model prediction
                predictions = model(train_sequence.to(device))
                labels = train_label.to(device)
                loss = criterion(predictions, labels)

                # Calculate accuracy and loss per batch
                correct = predictions.argmax(axis=1) == labels
                acc = correct.sum().item() / correct.size(0)
                train_set.set_postfix(loss=loss.item(), acc=acc)
                total_acc_train += correct.sum().item()
                total_loss_train += loss.item()

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            print(
                f'Epochs: {epoch + 1} | Loss: {total_loss_train / len(train_dataset): .3f} | Accuracy: {total_acc_train / len(train_dataset): .3f}')


def predict(test_dataset, model, vocab):
    data_text = test_dataset["Message"].values.tolist()
    labels = test_dataset["Category"].values.tolist()
    idx = 250
    seq = [vocab[word] for word in tokenizer(data_text[idx])]
    x = model(torch.tensor(seq).unsqueeze(0))
    print(data_text[idx])
    print(x.argmax(axis=1))
    print(label2id[labels[idx]])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv('spam_ham.csv')
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    print(df_train.head())

    labels = df_train["Category"].unique()
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    print(id2label)
    print(label2id)

    tokenizer = get_tokenizer('basic_english')
    epochs = 5
    lr = 1e-4
    batch_size = 4



    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)


    data_iter = TextIter(df_train)
    vocab = build_vocab_from_iterator(yield_tokens(data_iter), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    a = vocab.get_stoi()
    text = 'this is text'
    seq = [vocab[word] for word in tokenizer(text)]

    # print(tokenizer(text))
    # print(seq)

    model = Transformer(len(vocab), d_model=300, dff=50, h=4,
                        encoder_count=6).to(device)
    # model.load_state_dict(torch.load("model_version_1.pth"))
    train(model, df_train, epochs, lr, batch_size, vocab)
    torch.save(model.state_dict(), "model_version_1.pth")
    predict(df_test, model, vocab)


# emb = Embeddings(len(vocab), d_model)
# input_data = torch.LongTensor(seq).unsqueeze(0)
# token_emb = emb(input_data)
# # print(f'Size of token embedding: {token_emb.size()}')
# # print(token_emb)
# pos_encod = PositionalEncoding(d_model, len(vocab))
# pos_x = pos_encod(token_emb)
# att = Multihead(2, d_model)
# output_mult_att = att(pos_x)
# res_conn_1 = ResidualConnection(d_model=4)
# output_res_conn_1 = res_conn_1(pos_x, output_mult_att)
# ff = FeedForward(d_model, 12)
# x = ff(output_res_conn_1)
# res_conn_2 = ResidualConnection(d_model)
# new_x = res_conn_2(output_res_conn_1, x)
# print(x.shape)
# print(output_res_conn_1.shape)
# print(new_x.shape)

