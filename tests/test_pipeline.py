import csv, os, re, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import data_pipeline as D


def stats():
    return D.Stats()


# ---- steps 2-4: text cleaning
def test_strip_markup():
    assert D.strip_markup(r"{\an8}<i>Hello</i>\NWorld&nbsp;!") == "Hello World !"

def test_normalize_text_ascii_only():
    out = D.normalize_text("It\u2019s\u2026 \u201cfine\u201d \u2014 caf\u00e9")
    assert out == 'It\'s... "fine" - cafe'
    assert all(32 <= ord(c) < 127 for c in out)

def test_leading_dialogue_dash_removed():
    assert D.normalize_text("- Naruto!") == "Naruto!"

def test_clean_cue_drops_noise():
    s = stats()
    assert D.clean_cue("\u266a la la la \u266a", "X", s) is None
    assert D.clean_cue("(sighs)", "X", s) is None
    assert D.clean_cue("[music]", "X", s) is None
    assert D.clean_cue("...", "X", s) is None
    assert D.clean_cue("{\\an8}", "X", s) is None
    assert D.clean_cue("Believe it!", "X", s) == "Believe it!"
    assert D.clean_cue("(burn) we're gonna do it!", "X", s) is not None   # partial parentheses are kept


# ---- step 1: parsing
def test_parse_ep():
    cases = {
        "DBZ - 090 - Bold and Fearless.srt": 90,
        "DBZ - 120 - Super Saiyan 3 Arrives.srt": 120,
        "[HorribleSubs] Fairy Tail - 133 [720p].ass": 133,
        "Naruto Season 1 - 01.ass": 1,
        "Neon Genesis Evangelion EP (20).srt": 20,
        "[HorribleSubs] One Punch Man S2 - 09 [720p]_track3_eng.ass": 9,
        "no number here.ass": None,
    }
    for name, ep in cases.items():
        assert D.parse_ep(name) == ep, name

def test_read_ass_skips_signs_and_keeps_name(tmp_path):
    f = tmp_path / "x - 01.ass"
    f.write_text(
        "[Events]\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,Naruto,0,0,0,,Hello there\n"
        "Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,{\\pos(100,200)}SIGN TEXT\n"
        "Dialogue: 0,0:00:05.00,0:00:06.00,OP Song,,0,0,0,,la la\n"
        "Comment: 0,0:00:07.00,0:00:08.00,Default,,0,0,0,,ignored comment\n", encoding="utf-8")
    s = stats()
    rows = D.read_ass(str(f), "X", s)
    assert rows == [("Naruto", "Hello there")]
    assert s.c["X"]["01 typeset sign (\\pos/\\move)"] == 1
    assert s.c["X"]["01 sign/song style"] == 1

def test_load_folder_keeps_largest_file_per_episode(tmp_path):
    big = "1\n00:00:01,000 --> 00:00:02,000\nline one\n\n2\n00:00:03,000 --> 00:00:04,000\nline two\n\n"
    (tmp_path / "Show - 001.srt").write_text(big)
    (tmp_path / "Show - 001 [alt].srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nonly\n\n")
    (tmp_path / "Other - 002.srt").write_text(big)
    eps = D.load_folder(str(tmp_path), r"^Show", "Show", stats())
    assert list(eps) == [1] and len(eps[1]) == 2

def test_load_csv_sorts_and_drops_unnamed_episode(tmp_path):
    p = tmp_path / "op.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "start", "end", "character", "text"])
        w.writerow([400, "0:00:10.00", "0:00:11.00", "Luffy", "second"])
        w.writerow([400, "0:00:05.00", "0:00:06.00", "Zoro", "first"])
        w.writerow([401, "0:00:01.00", "0:00:02.00", "", "no name"])
        w.writerow([401, "0:00:03.00", "0:00:04.00", "", "no name 2"])
    s = stats()
    eps = D.load_csv(str(p), "OP", s)
    assert list(eps) == [400]
    assert [t for _, t in eps[400]] == ["first", "second"]
    assert s.c["OP"]["08 episode dropped: no character names"] == 1


# ---- steps 5-7
def test_consecutive_duplicates_removed():
    eps = {1: [(None, "Hi there!"), (None, "Hi there!"), (None, "Bye now!")]}
    out = D.clean_show(eps, "X", stats())
    assert [t for _, t in out[1]] == ["Hi there!", "Bye now!"]

def _fake_show(n_eps=10, n_lines=60):
    lyrics = ["we run through the night together", "chasing the light of tomorrow", "never stop believing in dreams"]
    eps = {}
    for ep in range(n_eps):
        rows = [(None, l) for l in lyrics]                                     # a lyric RUN at the start
        rows += [(None, f"unique filler line {ep}-{i} here") for i in range(n_lines)]
        rows.insert(5, (None, "Shadow clone jutsu!"))                          # lone catchphrase inside the edge region
        eps[ep] = rows
    return eps

def test_theme_run_removed_but_lone_catchphrase_kept():
    out, bad = D.remove_theme_lines(_fake_show(), "X", stats())
    texts = [t for _, t in out[0]]
    assert "we run through the night together" not in texts
    assert "Shadow clone jutsu!" in texts
    assert len(bad) == 3

def test_short_episodes_dropped():
    eps = {1: [(None, "x")] * 10, 2: [(None, "y")] * D.MIN_EP_LINES}
    assert list(D.drop_short_episodes(eps, "X", stats())) == [2]


# ---- steps 8-10
def test_parse_reply():
    got = D.parse_reply("0|Naruto\n1: Sasuke\n3|**Kakashi**\njunk", [0, 1, 2, 3])
    assert got == {0: "Naruto", 1: "Sasuke", 2: None, 3: "Kakashi"}

def test_label_lines_with_fake_client():
    class B:
        type = "text"
        def __init__(s, t): s.text = t
    class R:
        def __init__(s, t): s.content = [B(t)]
    class Msgs:
        calls = 0
        def create(s, **kw):
            s.calls += 1
            idx = [int(m.group(1)) for m in re.finditer(r"^(\d+): ", kw["messages"][0]["content"], re.M)]
            return R("\n".join(f"{i}|Naruto" for i in idx))
    c = types.SimpleNamespace(messages=Msgs())
    labs = D.label_lines(c, "m", "Naruto", 1, [f"line {i}" for i in range(200)])
    assert labs == ["Naruto"] * 200 and c.messages.calls == 3          # 200 lines / chunks of 80

def test_merge_fragments():
    s = stats()
    rows = [("A", "With its powerful tails,"), ("A", "it could smash mountains."),
            ("B", "Wow!"), ("A", "I was thinking..."), ("A", "maybe not."),
            ("A", "Done."), ("A", "Next.")]
    out = D.merge_fragments(rows, "X", s)
    assert out == [("A", "With its powerful tails, it could smash mountains."), ("B", "Wow!"),
                   ("A", "I was thinking... maybe not."), ("A", "Done."), ("A", "Next.")]

def test_merge_respects_max_length():
    long = "word " * 50
    out = D.merge_fragments([("A", long.strip()), ("A", long.strip())], "X", stats())
    assert len(out) == 2

def test_norm_speaker():
    n = D.norm_speaker
    assert n("NARUTO UZUMAKI") == "Naruto"
    assert n("MONKEY D. LUFFY") == "Luffy"
    assert n("Sasuke (thought)") == "Sasuke"
    assert n("Luffy & Zoro") == "Crowd"
    assert n("Kakashi:") == "Kakashi"
    assert n("") == "Unknown"
    assert n("Zabuza") == "Zabuza"
