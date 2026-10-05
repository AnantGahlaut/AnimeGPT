#!/usr/bin/env python3
"""
data_pipeline.py  -  ONE file that turns raw anime subtitle dumps into the training corpus.

=====================================================================================
WHERE THE DATA COMES FROM
=====================================================================================
 Naruto (original series, 220 eps)
   Kaggle  rakibulhasanshaon69/narutosubtitle  ("Naruto(subtitle )", Apache 2.0 per its metadata)
   218 .ass + 2 .srt. The .ass 'Name' field is empty on every line -> no speakers -> LLM-labeled.
 Dragon Ball Z (eps 1-100 are present: Saiyan + Namek sagas)
   Kaggle  jef1056/anime-subtitles  ("Anime Datasets V4"), folder  Extracted/Extracted/
   files look like  "DBZ - 090 - Bold and Fearless.srt". No speakers -> LLM-labeled.
   (The same dump has no Naruto Shippuden and no One Piece.)
 One Piece (eps 382-777, ~395 eps)
   Kaggle  ramazanturann/one-piece-transcripts-with-character-names-382-777
   (GitHub: MRamazan/One-Piece-Transcripts-with-Character-Names-382-777)
   CSV columns: episode,start,end,character,text. Fansub by Yibis Fansub; speakers are HUMAN-made.
   13 episodes have no character info (382,383,384,386,389,392,393,406,412,413,428,429,775) -> dropped.
 Licensing: all three are fan-made subtitles, for personal/educational use. Don't redistribute the corpus.

=====================================================================================
CLEANING STEPS (in order; every drop is counted and written to work/report.txt)
=====================================================================================
 1  PARSE        ASS 'Dialogue:' lines / SRT cues / CSV rows. ASS: skip typeset signs (\\pos, \\move),
                 sign/song/karaoke/credit styles. If an ASS file's Name field is filled, use it as speaker.
 2  STRIP MARKUP {\\an8...} override tags, <i> tags, \\N \\h newlines, &nbsp;
 3  NORMALIZE    curly quotes -> ', ... -> '...', dashes -> '-', accents stripped (e -> e), other non-ASCII dropped
 4  DROP NOISE   empty / punctuation-only lines, music-note lines, whole-line captions like "(sighs)" "[music]"
 5  DEDUPE       consecutive identical lines (overlapping cues, karaoke echo)
 6  THEME LINES  lines >=14 chars that sit in the first/last 15% of >=3% of a show's episodes AND form a run of
                 >=3 such lines (opening/ending lyrics, recaps, previews) are removed; lone catchphrases survive
 7  BROKEN EPS   episodes with < MIN_EP_LINES lines after cleaning are dropped
 8  SPEAKERS     One Piece: from the CSV (episodes without names dropped). Others: Claude labels every line
                 (cached per episode, re-labeled automatically if cleaning changed the lines)
 9  MERGE        subtitle cues split mid-sentence are joined when the same speaker keeps talking
 10 SPEAKER NORM aliases -> one canonical name, ALL CAPS -> Title, "A & B" -> Crowd, drop Sign/Unknown,
                 speakers with < MIN_SPEAKER_LINES lines -> Extra
 11 VALIDATE     only printable ASCII, no empty speaker/text, no ':' inside speaker names
 12 SPLIT        (done in training) hold out every 20th episode as validation, so no episode leaks

Output format (plain text, tokenizer-agnostic):
   <SHOW: Naruto>
   <EPISODE 1>
   Iruka: Naruto!
   Naruto: ...

USAGE
   pip install anthropic                      # only for LLM labeling
   export ANTHROPIC_API_KEY=sk-ant-...
   python3 data_pipeline.py \\
       --sub "Naruto=Subtitles" \\
       --sub "Dragon Ball Z=/path/Anime Datasets V4/Extracted/Extracted::(^|\\W)dbz\\W" \\
       --csv "One Piece=one_piece.csv" \\
       --label --out corpus.txt
   Add --limit 3 for a cheap test run. Without --label, shows that need labels are skipped (everything else builds).
   Read work/report.txt and work/review_sample.txt before training.
"""
import argparse, collections, csv, hashlib, json, os, random, re, statistics, sys, time, unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

WORK = "work"
MIN_EP_LINES = 40
MIN_SPEAKER_LINES = 25
THEME_EDGE_FRAC, THEME_MIN_LEN = 0.15, 14
CHUNK, CONTEXT = 80, 15
MAX_UTTERANCE = 220
SKIP_SPEAKERS = {"sign", "unknown"}
SIGN_STYLE_RX = re.compile(r"sign|song|karaoke|\bop\b|\bed\b|opening|ending|credit|title|staff|note|caption|translat", re.I)

SOURCES = [
    dict(show="Naruto", origin="Kaggle rakibulhasanshaon69/narutosubtitle", speakers="LLM"),
    dict(show="Dragon Ball Z", origin="Kaggle jef1056/anime-subtitles (Extracted/Extracted)", speakers="LLM"),
    dict(show="One Piece", origin="Kaggle ramazanturann/one-piece-transcripts-with-character-names-382-777", speakers="CSV (human)"),
]

ALIASES = {
    "naruto uzumaki": "Naruto", "sasuke uchiha": "Sasuke", "sakura haruno": "Sakura", "kakashi hatake": "Kakashi",
    "iruka umino": "Iruka", "hokage": "Third Hokage", "the third hokage": "Third Hokage", "lord third": "Third Hokage",
    "hinata hyuga": "Hinata", "neji hyuga": "Neji", "shikamaru nara": "Shikamaru", "choji akimichi": "Choji",
    "ino yamanaka": "Ino", "kiba inuzuka": "Kiba", "shino aburame": "Shino", "gaara of the sand": "Gaara",
    "lee": "Rock Lee", "guy": "Might Guy", "nine-tails": "Kurama", "nine tails": "Kurama", "kyuubi": "Kurama", "fox": "Kurama",
    "son goku": "Goku", "kakarot": "Goku", "son gohan": "Gohan", "king kai": "King Kai", "master roshi": "Master Roshi",
    "kame sennin": "Master Roshi", "captain ginyu": "Ginyu",
    "monkey d. luffy": "Luffy", "monkey d luffy": "Luffy", "roronoa zoro": "Zoro", "vinsmoke sanji": "Sanji",
    "tony tony chopper": "Chopper", "tony chopper": "Chopper", "nico robin": "Robin", "cutty flam": "Franky",
    "narration": "Narrator", "announcer": "Narrator", "on-screen text": "Sign", "text": "Sign", "sfx": "Sign",
    "all": "Crowd", "everyone": "Crowd", "multiple": "Crowd", "unknown speaker": "Unknown", "n/a": "Unknown", "": "Unknown",
}

ROSTERS = {
    "Naruto": "Naruto, Sasuke, Sakura, Kakashi, Iruka, Third Hokage, Tsunade, Jiraiya, Orochimaru, Kabuto, Shikamaru, Choji, Ino, Kiba, Hinata, Shino, Neji, Rock Lee, Tenten, Might Guy, Asuma, Kurenai, Gaara, Temari, Kankuro, Zabuza, Haku, Tazuna, Itachi, Kisame, Mizuki, Konohamaru, Ebisu, Anko, Ibiki, Gato, Kurama, Shizune, Pakkun, Narrator, Crowd, Sign, Unknown",
    "Dragon Ball Z": "Goku, Gohan, Piccolo, Krillin, Vegeta, Bulma, Chi-Chi, Master Roshi, Yamcha, Tien, Chiaotzu, Raditz, Nappa, Frieza, Zarbon, Dodoria, Dende, Nail, Guru, Ginyu, Recoome, Burter, Jeice, Guldo, King Kai, King Yemma, Mr. Popo, Kami, Oolong, Puar, Narrator, Crowd, Sign, Unknown",
}

SYSTEM = """You label subtitle lines from the anime {show} with who is speaking.

Input: a numbered list of subtitle lines (episode {ep}). Lines marked [context] are already labeled and shown only for continuity - do NOT output them.
Output: EXACTLY one line per unlabeled line, in order, formatted `<index>|<Speaker>`. No other text.

Rules:
- Use your knowledge of the show, this episode's plot, and the dialogue itself (who is addressed, catchphrases, what just happened).
- Prefer these canonical names: {roster}
- Characters not listed: use the name the show uses (first name or title). Never invent a long description.
- Narrator = series narrator / recap voice. Crowd = group voices. Sign = on-screen text, titles, sound effects. Unknown = genuinely cannot tell.
- Inner thoughts get the thinking character's name. A sentence split across several lines keeps the same speaker.
- Use ONE consistent spelling per character within the episode."""


class Stats:
    def __init__(self):
        self.c = collections.defaultdict(collections.Counter)   # show -> rule -> count
    def add(self, show, rule, n=1):
        self.c[show][rule] += n


# ------------------------------------------------------------------ steps 2-4: text cleaning
_NORM = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
                       "\u2015": "-", "\u2026": "...", "\u00a0": " ", "\u3000": " ", "\u2212": "-"})

def strip_markup(t):
    t = re.sub(r"\{[^}]*\}", "", t)
    t = re.sub(r"</?[^>]+>", "", t)
    t = t.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ").replace("&nbsp;", " ").replace("\n", " ")
    return t

def normalize_text(t):
    t = t.translate(_NORM)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = "".join(ch for ch in t if 32 <= ord(ch) < 127)
    t = re.sub(r"^\s*-\s*", "", t)              # leading dialogue dash
    return re.sub(r"\s+", " ", t).strip()

def clean_cue(raw, show, stats):
    """returns cleaned text or None (and counts why)"""
    if re.search(r"[\u266a\u266b\u266c]", raw):
        stats.add(show, "04 music-note line"); return None
    t = normalize_text(strip_markup(raw))
    if not t or not re.search(r"[A-Za-z0-9]", t):
        stats.add(show, "04 empty/punctuation-only"); return None
    if re.fullmatch(r"[\(\[][^\)\]]*[\)\]]", t):
        stats.add(show, "04 bracketed caption e.g. (sighs)"); return None
    return t


# ------------------------------------------------------------------ step 1: parsing
def parse_ep(name):
    s = os.path.splitext(name)[0]
    s = re.sub(r"\((\d{1,3})\)", r" \1 ", s)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"\([^)]*\)", " ", s)
    m = re.match(r"^\s*[^-]*?-\s*(\d{1,3})\s*-\s*\D", s)          # "DBZ - 090 - Title 2" -> 90
    if m:
        return int(m.group(1))
    nums = re.findall(r"(?<![\w.])(\d{1,3})(?:v\d)?(?![\w.])", s)
    return int(nums[-1]) if nums else None

def read_ass(path, show, stats):
    out = []
    for l in open(path, encoding="utf-8-sig", errors="ignore"):
        if not l.startswith("Dialogue:"):
            continue
        p = l.rstrip("\r\n").split(",", 9)
        if len(p) < 10:
            continue
        style, name, text = p[3], p[4].strip(), p[9]
        if SIGN_STYLE_RX.search(style):
            stats.add(show, "01 sign/song style"); continue
        if re.search(r"\\(pos|move)\(", text):
            stats.add(show, "01 typeset sign (\\pos/\\move)"); continue
        out.append((name or None, text))
    return out

def read_srt(path, show, stats):
    out = []
    for b in re.split(r"\r?\n\r?\n", open(path, encoding="utf-8-sig", errors="ignore").read()):
        lines = b.strip().splitlines()
        if len(lines) >= 3 and "-->" in lines[1]:
            out.append((None, " ".join(lines[2:])))
    return out

def load_folder(folder, name_rx, show, stats, limit=0):
    folder = os.path.expanduser(folder)
    files = [f for f in os.listdir(folder) if f.lower().endswith((".ass", ".ssa", ".srt"))]
    if name_rx:
        files = [f for f in files if re.search(name_rx, f, re.I)]
    best = {}
    for f in files:
        ep = parse_ep(f)
        if ep is None:
            stats.add(show, "01 file with no episode number"); continue
        sz = os.path.getsize(os.path.join(folder, f))
        if ep not in best or sz > best[ep][0]:
            best[ep] = (sz, f)
    eps = {}
    for ep in sorted(best)[: limit or None]:
        f = best[ep][1]
        reader = read_srt if f.lower().endswith(".srt") else read_ass
        eps[ep] = reader(os.path.join(folder, f), show, stats)
    return eps

def _ts(s):
    m = re.match(r"^(?:(\d+):)?(\d+):(\d+)(?:[.,](\d+))?$", str(s).strip())
    if not m:
        return None
    h, mi, se, fr = int(m.group(1) or 0), int(m.group(2)), int(m.group(3)), m.group(4) or "0"
    return h * 3600 + mi * 60 + se + int(fr) / 10 ** len(fr)

def load_csv(path, show, stats, limit=0):
    rows = list(csv.DictReader(open(os.path.expanduser(path), encoding="utf-8-sig", newline="")))
    if not rows:
        return {}
    cols = {k.lower().strip(): k for k in rows[0]}
    need = ["episode", "character", "text"]
    if any(n not in cols for n in need):
        sys.exit(f"CSV needs columns episode/character/text (+ optional start). Found: {list(rows[0])}")
    by_ep = collections.defaultdict(list)
    for i, r in enumerate(rows):
        try:
            ep = int(float(r[cols["episode"]]))
        except ValueError:
            continue
        st = _ts(r[cols["start"]]) if "start" in cols else None
        by_ep[ep].append((st, i, (r[cols["character"]] or "").strip() or None, r[cols["text"]] or ""))
    eps = {}
    for ep in sorted(by_ep)[: limit or None]:
        items = by_ep[ep]
        if all(x[0] is not None for x in items):
            items.sort(key=lambda x: (x[0], x[1]))
        named = sum(1 for x in items if x[2])
        if named < 0.5 * len(items):
            stats.add(show, "08 episode dropped: no character names"); continue
        eps[ep] = [(x[2], x[3]) for x in items]
    return eps


# ------------------------------------------------------------------ steps 2-7: per-show cleaning
def clean_show(eps, show, stats):
    cleaned = {}
    for ep, rows in eps.items():
        out, prev = [], None
        for sp, raw in rows:
            t = clean_cue(raw, show, stats)
            if t is None:
                continue
            if t == prev:
                stats.add(show, "05 consecutive duplicate"); continue
            prev = t
            out.append((sp, t))
        cleaned[ep] = out
    return cleaned

def remove_theme_lines(eps, show, stats):
    n = len(eps)
    thr = max(6, round(0.03 * n))
    seen = collections.defaultdict(set)
    def edge(i, L):
        e = max(10, int(L * THEME_EDGE_FRAC))
        return i < e or i >= L - e
    for ep, rows in eps.items():
        for i, (_, t) in enumerate(rows):
            if len(t) >= THEME_MIN_LEN and edge(i, len(rows)):
                seen[t.lower()].add(ep)
    bad = {k: len(v) for k, v in seen.items() if len(v) >= thr}
    out, dropped = {}, collections.Counter()
    for ep, rows in eps.items():
        L = len(rows)
        flag = [(t.lower() in bad and edge(i, L)) for i, (_, t) in enumerate(rows)]
        kill = [False] * L                      # lyrics come in RUNS; isolated catchphrases ("Shadow clone jutsu!") survive
        i = 0
        while i < L:
            if flag[i]:
                j = i
                while j < L and flag[j]:
                    j += 1
                if j - i >= 3:
                    for k in range(i, j):
                        kill[k] = True
                i = j
            else:
                i += 1
        keep = []
        for (sp, t), k in zip(rows, kill):
            if k:
                stats.add(show, "06 theme/recap line (run of >=3 repeated edge lines)")
                dropped[t.lower()] += 1
            else:
                keep.append((sp, t))
        out[ep] = keep
    bad = {k: v for k, v in bad.items() if k in dropped}
    return out, bad

def drop_short_episodes(eps, show, stats):
    out = {}
    for ep, rows in eps.items():
        if len(rows) < MIN_EP_LINES:
            stats.add(show, f"07 episode dropped (<{MIN_EP_LINES} lines)")
        else:
            out[ep] = rows
    return out


# ------------------------------------------------------------------ step 8: LLM speaker labels
def parse_reply(text, idxs):
    got = {}
    for l in text.splitlines():
        m = re.match(r"^\s*(\d+)\s*[|:]\s*(.+?)\s*$", l)
        if m:
            got[int(m.group(1))] = m.group(2).strip().strip("*[]")
    return {i: got.get(i) for i in idxs}

def label_lines(client, model, show, ep, lines):
    roster = ROSTERS.get(show, "Narrator, Crowd, Sign, Unknown")
    labels, seen = [None] * len(lines), collections.Counter()
    for start in range(0, len(lines), CHUNK):
        idxs = list(range(start, min(start + CHUNK, len(lines))))
        body = [f"[context] {i}|{labels[i]}: {lines[i]}" for i in range(max(0, start - CONTEXT), start)]
        body += [f"{i}: {lines[i]}" for i in idxs]
        user = ("Speakers seen so far in this episode: " + ", ".join(n for n, _ in seen.most_common(20)) + "\n\n" if seen else "") + "\n".join(body)
        res = None
        for attempt in range(4):
            try:
                r = client.messages.create(model=model, max_tokens=2000, temperature=0,
                                           system=SYSTEM.format(show=show, ep=ep, roster=roster),
                                           messages=[{"role": "user", "content": user}])
                res = parse_reply("".join(b.text for b in r.content if b.type == "text"), idxs)
                if sum(v is None for v in res.values()) <= 0.1 * len(idxs):
                    break
            except Exception as e:
                print(f"  {show} ep {ep}: {type(e).__name__}, retry {attempt + 1}", flush=True)
                time.sleep(2 * (attempt + 1) ** 2)
        res = res or {i: None for i in idxs}
        for i in idxs:
            labels[i] = res[i] or "Unknown"
            seen[labels[i]] += 1
    return labels

def label_show(eps, show, client, model, workers, stats):
    slug = re.sub(r"[^a-z0-9]+", "_", show.lower())
    cache = os.path.join(WORK, "labels", slug)
    os.makedirs(cache, exist_ok=True)
    def job(ep):
        lines = [t for _, t in eps[ep]]
        h = hashlib.sha1("\n".join(lines).encode()).hexdigest()
        f = os.path.join(cache, f"ep_{ep:04d}.json")
        if os.path.exists(f):
            d = json.load(open(f))
            if d.get("hash") == h:
                return ep, d["labels"]
        labs = label_lines(client, model, show, ep, lines)
        json.dump({"hash": h, "labels": labs}, open(f, "w"))
        return ep, labs
    out = {}
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(job, ep) for ep in eps]
        for k, fu in enumerate(as_completed(futs), 1):
            ep, labs = fu.result()
            out[ep] = [(lab, t) for lab, (_, t) in zip(labs, eps[ep])]
            print(f"  [{show}] labeled {k}/{len(eps)}", flush=True)
    return out


# ------------------------------------------------------------------ steps 9-10: merge + speaker normalisation
def merge_fragments(rows, show, stats):
    out = []
    for sp, t in rows:
        if out:
            psp, pt = out[-1]
            ends = re.search(r"[.!?][\"')\]]*$", pt) is not None
            if pt.endswith("..."):
                ends = t[:1].isupper()                       # "..." then lowercase = sentence continues
            if psp == sp and not ends and len(pt) + len(t) + 1 <= MAX_UTTERANCE:
                out[-1] = (sp, pt + " " + t)
                stats.add(show, "09 cue merged into previous utterance")
                continue
        out.append((sp, t))
    return out

def norm_speaker(name):
    n = re.sub(r"\s*\((thought|thinking|v\.?o\.?|off-?screen|o\.?s\.?)\)\s*$", "", name or "", flags=re.I)
    n = re.sub(r"\s+", " ", n.replace(":", " ")).strip(" -_.")
    if re.search(r"\s(and)\s|[&/,+]", n):
        return "Crowd"
    if n.isupper() and len(n) > 3:
        n = n.title()
    return ALIASES.get(n.lower(), n)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", action="append", default=[], help='"Show=folder" or "Show=folder::filename-regex" (.ass/.srt)')
    ap.add_argument("--csv", action="append", default=[], help='"Show=file.csv" with columns episode,character,text[,start]')
    ap.add_argument("--out", default="corpus.txt")
    ap.add_argument("--label", action="store_true", help="LLM-label shows that have no speaker names (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001", help="try claude-sonnet-5-5 for better labels")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="only first N episodes per show (test run)")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    random.seed(0)
    stats = Stats()

    raw = {}   # show -> {ep: [(speaker|None, text)]}
    for s in a.sub:
        show, _, rest = s.partition("=")
        folder, _, rx = rest.partition("::")
        raw[show.strip()] = load_folder(folder, rx or None, show.strip(), stats, a.limit)
    for s in a.csv:
        show, _, path = s.partition("=")
        raw[show.strip()] = load_csv(path, show.strip(), stats, a.limit)
    if not raw:
        sys.exit("nothing to do: pass --sub and/or --csv")

    client = None
    final = {}   # show -> {ep: [(speaker, text)]}
    theme_report = {}
    for show, eps in raw.items():
        print(f"== {show}: {len(eps)} episodes parsed")
        eps = clean_show(eps, show, stats)
        eps, bad = remove_theme_lines(eps, show, stats)
        theme_report[show] = bad
        eps = drop_short_episodes(eps, show, stats)
        total = sum(len(r) for r in eps.values())
        named = sum(1 for r in eps.values() for sp, _ in r if sp)
        if total and named >= 0.6 * total:
            labeled = {ep: [(sp or "Unknown", t) for sp, t in rows] for ep, rows in eps.items()}
            print(f"   speakers come from the source ({named}/{total} lines named)")
        elif a.label:
            if client is None:
                import anthropic
                client = anthropic.Anthropic()
            labeled = label_show(eps, show, client, a.model, a.workers, stats)
        else:
            print(f"   !! {show} has no speaker names and --label was not given -> SKIPPED")
            continue
        final[show] = labeled

    # step 9-10: merge, normalise
    for show in final:
        for ep in final[show]:
            rows = [(norm_speaker(sp), t) for sp, t in final[show][ep]]
            rows = merge_fragments(rows, show, stats)
            kept = [(sp, t) for sp, t in rows if sp.lower() not in SKIP_SPEAKERS]
            stats.add(show, "10 Sign/Unknown line dropped", len(rows) - len(kept))
            final[show][ep] = kept
    freq = collections.Counter(sp for sh in final.values() for rows in sh.values() for sp, _ in rows)
    rare = {sp for sp, c in freq.items() if c < MIN_SPEAKER_LINES}

    # step 11: validate + write
    lines_out, n_lines, cl = 0, 0, []
    spk = collections.defaultdict(collections.Counter)
    with open(a.out, "w", encoding="utf-8") as f:
        for show, eps in final.items():
            for ep in sorted(eps):
                rows = [("Extra" if sp in rare else sp, t) for sp, t in eps[ep]]
                if not rows:
                    continue
                f.write(f"<SHOW: {show}>\n<EPISODE {ep}>\n")
                for sp, t in rows:
                    assert sp and t and ":" not in sp, (show, ep, sp, t)
                    assert all(32 <= ord(c) < 127 for c in sp + t), (show, ep, sp, t)
                    f.write(f"{sp}: {t}\n")
                    n_lines += 1; cl.append(len(t)); spk[show][sp] += 1
                f.write("\n")
                eps[ep] = rows
    chars = sorted(set(open(a.out, encoding="utf-8").read()))

    # reports
    rep = ["SOURCES"] + [f"  {s['show']}: {s['origin']} | speakers: {s['speakers']}" for s in SOURCES]
    rep += ["", f"CORPUS {a.out}: {n_lines} lines, {os.path.getsize(a.out)/1e6:.2f} MB, {len(chars)} distinct chars",
            f"  utterance length chars: median {statistics.median(cl):.0f}, p95 {sorted(cl)[int(.95*len(cl))]}, max {max(cl)}" if cl else "  (empty)",
            "  charset: " + "".join(chars).replace("\n", "\\n"), ""]
    for show in raw:
        rep.append(f"== {show}  (episodes kept: {len(final.get(show, {}))})")
        for rule, c in sorted(stats.c[show].items()):
            rep.append(f"   {c:7}  {rule}")
        if show in spk:
            rep.append("   speakers: " + ", ".join(f"{s} {c}" for s, c in spk[show].most_common(30)))
        bad = theme_report.get(show, {})
        if bad:
            rep.append(f"   theme/recap lines removed ({len(bad)} distinct) - READ THIS LIST for false positives:")
            rep += [f"      {n:3} eps | {t}" for t, n in sorted(bad.items(), key=lambda x: -x[1])[:60]]
        rep.append("")
    rep.append(f"rare speakers folded into 'Extra' (<{MIN_SPEAKER_LINES} lines): {len(rare)}")
    open(os.path.join(WORK, "report.txt"), "w", encoding="utf-8").write("\n".join(rep))

    rev = []
    for show, eps in final.items():
        pool = [(ep, i) for ep, rows in eps.items() for i in range(len(rows))]
        for ep, i in random.sample(pool, min(40, len(pool))):
            rows = eps[ep]
            rev.append(f"--- {show} ep {ep} line {i}")
            for j in range(max(0, i - 2), min(len(rows), i + 3)):
                rev.append(("  >> " if j == i else "     ") + f"{rows[j][0]}: {rows[j][1]}")
    open(os.path.join(WORK, "review_sample.txt"), "w", encoding="utf-8").write("\n".join(rev))
    print("\n".join(rep[:12]))
    print(f"\nwrote {a.out}, {WORK}/report.txt, {WORK}/review_sample.txt")


if __name__ == "__main__":
    main()
