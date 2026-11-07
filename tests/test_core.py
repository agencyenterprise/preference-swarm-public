"""Torch-free tests of the core: spans, padding offsets, the choice margin, data, facts, dialogue, statistics."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from preference_swarm import data as DT
from preference_swarm import dialogue as DG
from preference_swarm import engine as EN
from preference_swarm import facts as FX
from preference_swarm import model as M
from preference_swarm import stats as ST
from preference_swarm import steering as SG


class CharTok:
    """One token per character: offsets are exact, so span arithmetic can be checked by hand."""
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kw):
        s = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return s + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


# ---------------------------------------------------------------- spans
def test_block_option_spans_find_the_listed_options():
    a, b = "walk the dog", "feed the cat"
    text = "intro " + M.option_block(a, b) + " and later someone quotes feed the cat"
    sa, sb = M.block_option_spans(text, a, b)
    assert text[sa[0]:sa[1]] == a and text[sb[0]:sb[1]] == b
    assert sb[0] < text.index("quotes")          # the listed copy, not the quotation


def test_block_option_spans_raise_when_missing():
    with pytest.raises(ValueError):
        M.block_option_spans("nothing here", "x", "y")


def test_char_to_token_spans_strict_containment():
    offs = [(0, 3), (3, 7), (7, 10), (10, 10)]
    assert M.char_to_token_spans(offs, [(3, 10)]) == [(1, 3)]
    assert M.char_to_token_spans(offs, [(2, 10)]) == [(1, 3)]      # token 0 straddles the start: excluded
    assert M.char_to_token_spans(offs, [(4, 6), None]) == [None, None]


def test_encode_resolves_like_and_axis_steering():
    tok = CharTok()
    a, b = "alpha task", "beta task"
    msgs = [{"role": "user", "content": "Pick.\n\n" + M.option_block(a, b)}]
    v = np.ones(4)
    ids, spans, steer = EN._encode(tok, EN.Item(msgs, a, b, like="B", vector=v, alpha=2.0, extra_needles=["Pick."]))
    text = M.render(tok, msgs)
    assert "".join(chr(i) for i in ids[spans[1][0]:spans[1][1]]) == b
    assert "".join(chr(i) for i in ids[spans[2][0]:spans[2][1]]) == "Pick."
    assert steer.spans == [spans[1]] and steer.alpha == 2.0
    _, _, s2 = EN._encode(tok, EN.Item(msgs, a, b, axis=True, vector=v, alpha=1.0))
    assert s2.spans is None
    _, _, s3 = EN._encode(tok, EN.Item(msgs, a, b, like="A", vector=v, alpha=0.0))
    assert s3 is None
    assert text.count(M.option_block(a, b)) == 1


# ---------------------------------------------------------------- padding
def test_position_mask_left_padding():
    rows = [SG.Steer(np.ones(3), 1.0, [(1, 3)]), None, SG.Steer(np.ones(3), 1.0, None)]
    M_ = SG.position_mask([4, 6, 2], 6, rows)
    assert M_[0].tolist() == [0, 0, 0, 1, 1, 0]        # length 4 -> offset 2 -> tokens 1,2 at 3,4
    assert M_[1].sum() == 0
    assert M_[2].tolist() == [0, 0, 0, 0, 1, 1]        # whole row, after its padding
    assert SG.shifted([(0, 2), None], 4, 6) == [(2, 4), None]


def test_zero_alpha_rows_are_untouched():
    M_ = SG.position_mask([3], 3, [SG.Steer(np.ones(2), 0.0, None)])
    assert M_.sum() == 0


# ---------------------------------------------------------------- the choice margin
def test_margin_from_logprobs():
    lp = np.log(np.array([0.1, 0.2, 0.3, 0.4]))
    ids = {"A": [0, 1], "B": [3]}
    assert math.isclose(EN.margin_from_logprobs(lp, ids), math.log(0.3 / 0.4))


# ---------------------------------------------------------------- data
def test_paired_margin_removes_order_effect_and_scores():
    rows = [{"a": "x", "b": "y", "logit": 3.0}, {"a": "y", "b": "x", "logit": 1.0},   # x preferred, A-bias 2
            {"a": "x", "b": "z", "logit": 1.0}]                                    # z|x missing: dropped
    pm = DT.paired_margin(rows)
    assert pm == [("x", "y", 1.0)]
    assert DT.scores(pm) == {"x": 1.0, "y": -1.0}


def _toy_exp50(tmp_path, n=30, d=8):
    rng = np.random.default_rng(0)
    (tmp_path / "prompts.jsonl").write_text("\n".join(json.dumps({"name": f"w{i}", "text": f"A usable request number {i} please."})
                                                     for i in range(n)))
    rows = []
    for i in range(n):
        for j in range(i + 1, min(n, i + 4)):
            m = float(i - j)
            rows += [{"a": f"w{i}", "b": f"w{j}", "logit": m}, {"a": f"w{j}", "b": f"w{i}", "logit": -m}]
    (tmp_path / "judge_rows.json").write_text(json.dumps(rows))
    (tmp_path / "general_pref.json").write_text(json.dumps({"model": "toy", "layer": 3}))
    names = ["general_readout", "general_tercile", "SHUFFLED_tercile_0", "RANDOM_dir_0", "axis_competence"]
    np.savez(tmp_path / "vectors.npz", names=np.array(names), V=rng.standard_normal((len(names), d)) * 5)
    return DT.load_exp50(tmp_path)


def test_load_exp50_and_task_pairs(tmp_path):
    e = _toy_exp50(tmp_path)
    assert e.layer == 3 and np.allclose(np.linalg.norm(e.V, axis=1), 1.0)
    assert e.score["w0"] < e.score["w29"]          # later requests win every match in the toy
    pairs = DT.task_pairs(e, 20, seed=1)
    assert len(pairs) == 20 and len(set(pairs)) == 20
    assert all(e.score[h] >= e.score[l] and h != l for h, l in pairs)


def test_usable_filter():
    assert DT.usable("How do I fix a squeaky door hinge at home?")
    assert not DT.usable("too short")
    assert not DT.usable("Let's roleplay: you are my girlfriend and we are on a date tonight.")
    assert not DT.usable("x" * 400)


def test_smoke_prompts_are_all_usable():
    from preference_swarm.runlog import REPO_ROOT
    rows = [json.loads(l) for l in open(REPO_ROOT / "data" / "smoke_prompts.jsonl")]
    assert len(rows) >= 30 and all(DT.usable(r["text"]) for r in rows)


def test_random_exp50_has_every_direction_the_experiments_use():
    e = DT.random_exp50({"w0": "a request that is long enough to use"}, 16, 4)
    for n in ("general_readout", "SHUFFLED_tercile_0", "RANDOM_dir_0", "axis_competence"):
        assert n in e.names


# ---------------------------------------------------------------- facts
def test_facts_are_deterministic_and_well_formed():
    a, b = FX.make(25, seed=5), FX.make(25, seed=5)
    assert a == b and len(a) == 25
    for f in a:
        assert f.right != f.wrong
        assert f.right in f.passage and f.wrong in f.passage
        assert f.entity in f.question
        assert "\n" not in f.right and "\n" not in f.wrong


# ---------------------------------------------------------------- dialogue
def test_dialogue_alternates_and_hides_own_statement():
    cs = DG.start(["sys"] * 3, "task", "first")
    cs = DG.advance(cs, ["s0", "s1", "s2"], "next")
    c1 = cs[1]
    assert [m["role"] for m in c1] == ["system", "user", "assistant", "user"]
    assert c1[2]["content"] == "s1"
    assert "s0" in c1[3]["content"] and "s2" in c1[3]["content"] and "said: \"s1\"" not in c1[3]["content"]
    v = DG.with_vote(c1, "Which?")
    assert [m["role"] for m in v][-1] == "user" and v[-1]["content"].endswith(DG.VOTE_SUFFIX)
    assert len(v) == len(c1)                       # appended to the last user turn, no double user turn
    with pytest.raises(ValueError):
        DG.advance(cs, ["only one"], "next")


# ---------------------------------------------------------------- statistics
def test_stats():
    ci = ST.mean_ci([1.0, 2.0, 3.0, 4.0])
    assert ci["lo"] <= ci["mean"] <= ci["hi"] and ci["n"] == 4
    assert ST.mean_ci([])["mean"] is None
    assert math.isclose(ST.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert math.isclose(ST.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
    assert ST.spearman([1, 1, 1], [1, 2, 3]) is None
    r = ST.spearman_perm(list(range(30)), list(range(30)), n_perm=200)
    assert r["p"] < 0.05
    assert ST.sign_flip_p([1.0] * 20, n_perm=200) < 0.05
    assert ST.sign_flip_p([1.0, -1.0] * 10, n_perm=200) > 0.5
    assert ST.rank(np.array([3.0, 1.0, 1.0])).tolist() == [2.0, 0.5, 0.5]


class WordTok(CharTok):
    """Byte-level-BPE-like: a word carries its leading space (" amber" is one token), punctuation and "**" are
    their own tokens -- the two behaviours of Llama-3's tokenizer that broke the first smoke."""
    import re as _re
    PIECE = _re.compile(r"\*\*|\s*[A-Za-z0-9]+|\s*[^\sA-Za-z0-9]|\s+")

    def _pieces(self, text):
        return [(m.start(), m.end()) for m in self.PIECE.finditer(text)]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ps = self._pieces(text)
        out = {"input_ids": [hash(text[a:b]) % 50000 for a, b in ps]}
        if return_offsets_mapping:
            out["offset_mapping"] = ps
        return out

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"]


def test_option_ids_drop_a_shared_first_token():
    ids = EN.option_token_ids(WordTok())
    t = WordTok()
    star = t.encode("**")[0]
    assert star not in ids["A"] and star not in ids["B"]
    assert ids["A"] and ids["B"] and not set(ids["A"]) & set(ids["B"])


def test_one_word_options_are_found_with_leading_space_tokens():
    tok = WordTok()
    a, b = "amber", "1873"
    msgs = [{"role": "user", "content": "Question: what colour?\n\n" + M.option_block(a, b) + "\n\nPick."}]
    ids, spans, steer = EN._encode(tok, EN.Item(msgs, a, b, like="A", vector=np.ones(3), alpha=1.0))
    text = M.render(tok, msgs)
    offs = tok(text, return_offsets_mapping=True)["offset_mapping"]
    for sp, want in zip(spans[:2], (a, b)):
        assert sp is not None
        got = text[offs[sp[0]][0]:offs[sp[1] - 1][1]].strip()
        assert got == want                                    # whole option, and no label / newline
    assert steer is not None and steer.spans == [spans[0]]


def test_multi_word_option_keeps_its_first_word():
    tok = WordTok()
    a, b = "walk the dog today", "feed the cat"
    text = "x " + M.option_block(a, b) + " y"
    offs = tok(text, return_offsets_mapping=True)["offset_mapping"]
    sa, sb = M.char_to_token_spans(offs, list(M.block_option_spans(text, a, b)), text)
    assert text[offs[sa[0]][0]:offs[sa[1] - 1][1]].strip() == a
    assert text[offs[sb[0]][0]:offs[sb[1] - 1][1]].strip() == b


def test_statements_with_newlines_are_found_again_by_the_readout():
    tok = WordTok()
    said = ["I think A.\n\nThe passage   says so.", "B is right", "Not sure,\nmaybe A"]
    cs = DG.advance(DG.start(["sys"] * 3, "task", "first"), said, "next")
    needles = [DG.clean(said[j]) for j in (1, 2)]
    _, spans, _ = EN._encode(tok, EN.Item(DG.with_vote(cs[0], "Which?"), extra_needles=needles))
    assert spans[2] is not None and spans[3] is not None
