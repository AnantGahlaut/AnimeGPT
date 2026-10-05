"""
Tiny GPT (decoder-only transformer, char-level) trained on the labeled anime corpus,
plus an interactive chat where you speak as one character and the model answers as another.

  pip install torch
  python3 gpt.py train --data corpus.txt --small --iters 200      # 1-min smoke test (works on CPU)
  python3 gpt.py train --data corpus.txt                           # real run (wants a GPU)
  python3 gpt.py chat  --ckpt ckpt.pt --show Naruto

Paper map ("Attention Is All You Need"):
  CausalSelfAttention -> 3.2.1 scaled dot-product + 3.2.2 multi-head (decoder mask = causal)
  Block               -> residual + LayerNorm around attention and the feed-forward net (3.1, 3.3)
  GPT.pos_emb         -> 3.5 positions (learned here instead of sinusoids)
  We use ONLY the decoder stack (no encoder / cross-attention): that's what makes it a "GPT".
"""
import argparse, math, os, random, re, sys, time


# ---------------------------------------------------------------- data helpers (no torch needed)
def split_chunks(text):
    """split corpus into per-episode chunks, each starting with '<SHOW: ...>'"""
    return [c for c in re.split(r"(?=<SHOW: )", text) if c.strip()]

def train_val_split(text, every=20):
    chunks = split_chunks(text)
    val = [c for i, c in enumerate(chunks) if i % every == every - 1]
    tr = [c for i, c in enumerate(chunks) if i % every != every - 1]
    return "".join(tr), "".join(val)

def build_vocab(text):
    chars = sorted(set(text))
    return chars, {c: i for i, c in enumerate(chars)}

def encode(s, stoi):
    return [stoi[c] for c in s if c in stoi]      # unseen chars are dropped

def decode(ids, chars):
    return "".join(chars[i] for i in ids)

def make_header(show, ep=1):
    return f"<SHOW: {show}>\n<EPISODE {ep}>\n"

def parse_user_line(raw, me):
    """'Sasuke: NARUTO!!!!' -> ('Sasuke', 'NARUTO!!!!'); plain text is spoken by `me`"""
    m = re.match(r"^\s*([A-Z][A-Za-z0-9 .'\-]{0,24}):\s*(.+)$", raw)
    if m and len(m.group(1).split()) <= 3:
        return m.group(1).strip(), m.group(2).strip()
    return me, raw.strip()


# ---------------------------------------------------------------- model
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)   # W_Q, W_K, W_V for all heads at once
        self.proj = nn.Linear(n_embd, n_embd)                  # W_O
        self.drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        d_k = C // self.n_head
        q = q.view(B, T, self.n_head, d_k).transpose(1, 2)     # (B, heads, T, d_k)
        k = k.view(B, T, self.n_head, d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, d_k).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)       # (B, heads, T, T) similarity of every token pair
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))   # can't look at the future
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)  # weighted mix of values, heads concatenated
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        # pre-LayerNorm (GPT-2 style); the original paper put the norm after the residual add
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.head.weight = self.tok_emb.weight                  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


@torch.no_grad()
def generate_line(model, ids, chars, stoi, temperature=0.8, top_k=40, max_new=240):
    """extend `ids` (list[int]) until the model emits a newline; return the new text (without the newline)"""
    model.eval()
    dev = next(model.parameters()).device
    nl = stoi["\n"]
    out = []
    for _ in range(max_new):
        ctx = torch.tensor([(ids + out)[-model.block_size:]], dtype=torch.long, device=dev)
        logits, _ = model(ctx)
        logits = logits[0, -1] / max(temperature, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        if nxt == nl:
            break
        out.append(nxt)
    return decode(out, chars)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------- train
def cmd_train(a):
    random.seed(1337); torch.manual_seed(1337)
    dev = pick_device()
    text = open(a.data, encoding="utf-8").read()
    chars, stoi = build_vocab(text)
    tr_txt, va_txt = train_val_split(text)
    tr = torch.tensor(encode(tr_txt, stoi), dtype=torch.long)
    va = torch.tensor(encode(va_txt, stoi), dtype=torch.long)
    print(f"device={dev} vocab={len(chars)} train={len(tr):,} val={len(va):,} chars")

    cfg = dict(vocab=len(chars), block_size=a.block, n_layer=a.layers, n_head=a.heads, n_embd=a.embd, dropout=a.dropout)
    if a.small:
        cfg.update(block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.1)
        a.batch = min(a.batch, 32)
    model = GPT(**cfg).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  config: {cfg}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    bs = cfg["block_size"]

    def batch(split):
        d = tr if split == "train" else va
        ix = torch.randint(len(d) - bs - 1, (a.batch,))
        x = torch.stack([d[i:i + bs] for i in ix]); y = torch.stack([d[i + 1:i + bs + 1] for i in ix])
        return x.to(dev), y.to(dev)

    @torch.no_grad()
    def evaluate():
        model.eval(); res = {}
        for split in ("train", "val"):
            ls = []
            for _ in range(a.eval_batches):
                x, y = batch(split)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    ls.append(model(x, y)[1].item())
            res[split] = sum(ls) / len(ls)
        model.train(); return res

    def lr_at(it):
        warm = min(100, a.iters // 10)
        if it < warm:
            return a.lr * (it + 1) / warm
        p = (it - warm) / max(1, a.iters - warm)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    best, bad, t0 = 1e9, 0, time.time()
    shows = re.findall(r"<SHOW: ([^>]+)>", text)
    demo_show = shows[0] if shows else "Naruto"
    for it in range(a.iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        if it % a.eval_every == 0:
            r = evaluate()
            print(f"iter {it:5d} | train {r['train']:.3f} | val {r['val']:.3f} | {time.time()-t0:.0f}s", flush=True)
            if r["val"] < best:
                best, bad = r["val"], 0
                torch.save({"model": model.state_dict(), "cfg": cfg, "chars": chars}, a.ckpt)
            else:
                bad += 1
                if bad >= a.patience:
                    print(f"val stopped improving for {bad} evals -> stopping (best val {best:.3f} saved to {a.ckpt})")
                    break
            if it:
                ids = encode(make_header(demo_show), stoi)
                print("   sample>", generate_line(model, ids, chars, stoi)[:160])
        x, y = batch("train")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    print(f"done. best val loss {best:.3f}. checkpoint: {a.ckpt}")


# ---------------------------------------------------------------- chat
def cmd_chat(a):
    dev = pick_device()
    ck = torch.load(a.ckpt, map_location=dev)
    chars = ck["chars"]; stoi = {c: i for i, c in enumerate(chars)}
    model = GPT(**ck["cfg"]).to(dev); model.load_state_dict(ck["model"]); model.eval()
    show, me, forced, n_replies, temp = a.show, a.me, None, 1, a.temp
    hist = []
    print("Type 'Sasuke: NARUTO!!!!' to speak as a character (plain text = spoken by --me).")
    print("Commands: /show NAME  /reply NAME|auto  /n K  /temp T  /reset  /me NAME  /quit\n")

    def prompt_ids(extra=""):
        return encode(make_header(show) + "".join(l + "\n" for l in hist) + extra, stoi)

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        if raw.startswith("/"):
            cmd, _, arg = raw[1:].partition(" ")
            arg = arg.strip()
            if cmd == "quit": break
            elif cmd == "reset": hist = []
            elif cmd == "show" and arg: show, hist = arg, []
            elif cmd == "me" and arg: me = arg
            elif cmd == "reply": forced = None if arg.lower() == "auto" or not arg else arg
            elif cmd == "n" and arg.isdigit(): n_replies = max(1, int(arg))
            elif cmd == "temp":
                try: temp = float(arg)
                except ValueError: pass
            else: print("?")
            print(f"[show={show} me={me} reply={forced or 'auto'} n={n_replies} temp={temp}]")
            continue
        who, text = parse_user_line(raw, me)
        hist.append(f"{who}: {text}")
        for _ in range(n_replies):
            line = ""
            for _try in range(6):
                if forced:
                    gen = generate_line(model, prompt_ids(f"{forced}:"), chars, stoi, temp)
                    cand = f"{forced}:{gen}"
                else:
                    cand = generate_line(model, prompt_ids(), chars, stoi, temp)
                if cand.strip() and not cand.startswith("<") and ":" in cand:
                    line = cand; break
            if not line:
                print("(model produced nothing usable, try again or raise /temp)"); break
            hist.append(line)
            print(line)
        hist = hist[-40:]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="corpus.txt"); t.add_argument("--ckpt", default="ckpt.pt")
    t.add_argument("--small", action="store_true", help="tiny model for CPU smoke tests")
    t.add_argument("--iters", type=int, default=5000); t.add_argument("--batch", type=int, default=64)
    t.add_argument("--block", type=int, default=256); t.add_argument("--layers", type=int, default=6)
    t.add_argument("--heads", type=int, default=6); t.add_argument("--embd", type=int, default=384)
    t.add_argument("--dropout", type=float, default=0.2); t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--eval-every", type=int, default=250, dest="eval_every")
    t.add_argument("--eval-batches", type=int, default=20, dest="eval_batches")
    t.add_argument("--patience", type=int, default=6)
    c = sub.add_parser("chat")
    c.add_argument("--ckpt", default="ckpt.pt"); c.add_argument("--show", default="Naruto")
    c.add_argument("--me", default="Naruto"); c.add_argument("--temp", type=float, default=0.8)
    a = ap.parse_args()
    {"train": cmd_train, "chat": cmd_chat}[a.cmd](a)


if __name__ == "__main__":
    main()
