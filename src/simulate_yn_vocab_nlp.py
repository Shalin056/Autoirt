"""
simulate_yn_vocab_nlp.py
==========================

First-pass replacement for simulate_det_items' Y/N Vocab generation.
Where the old version gave every item two arbitrary uniform random
numbers and defined difficulty/discrimination as an invented sine/cosine
function of them, this generates REAL word content, computes REAL
linguistic features from it, and defines difficulty from a relationship
the literature actually supports, rather than a function nobody has ever
observed in real data.

WHY THIS DESIGN, SPECIFICALLY

Word frequency: every single paper in this week's literature review that
targets vocabulary-adjacent items treats frequency as a, usually THE,
dominant predictor of difficulty. This is also the field's own dominant
feature category (AlKhuzaey et al. 2024's review: 54% hand-crafted
linguistic features across 55 studies, frequency chief among them) --
not a fringe choice. Uses the `wordfreq` package (pip install wordfreq),
which ships real corpus-derived frequency data offline, no network
needed, on a Zipf scale (same underlying concept as the paper's own
COCA-derived log frequency).

Real vs. fake words need DIFFERENT difficulty logic, not the same
formula applied to both, which is what distinguishes this from a naive
port of "frequency predicts difficulty":
  - For REAL words, difficulty should increase as frequency DECREASES.
    Rarer words are harder to correctly recognize as real. This is the
    frequency effect, one of the most replicated findings in lexical
    decision research.
  - For FAKE words, difficulty should increase as "wordlikeness"
    INCREASES. A fake word that looks/sounds like a plausible English
    word (right letter patterns, right structure) is harder to
    correctly reject than an obviously fake one. This is the
    orthographic neighborhood / wordlikeness effect, the other classic
    finding in the same literature. Fake words never appear in a
    frequency corpus (frequency 0 for all of them), so frequency alone
    gives zero signal for this half of the item bank -- wordlikeness,
    approximated here via character n-gram frequency against the real
    corpus, is what actually varies among fake words and is exactly the
    kind of feature the AutoIRT paper itself describes computing
    (n-gram prefix/suffix frequency, n-grams over a threshold, etc.).

Fake words are generated with a simple order-2 character Markov model
trained on the real corpus, not an RNN like the paper's, which is a
real simplification worth being upfront about -- but it produces
phonotactically plausible non-words (right letter transition
statistics) using only what's available offline, and gives fake words a
genuine SPREAD of wordlikeness scores rather than all being equally
fake, which is what the difficulty relationship above actually needs to
have something to bite into.

DISCRIMINATION IS DELIBERATELY LEFT WEAK AND NOISY, NOT CONFIDENTLY
MODELED. This week's literature search turned up almost nothing on what
predicts discrimination from content features -- R2DE (Benedetto et al.
2020) and Byrd & Srivastava (2022) are the only two examples found, and
both report discrimination as harder to predict than difficulty from
the same features. Inventing a confident discrimination formula here
would just be swapping one made-up function (the old sine/cosine one)
for a different made-up function dressed in real features. Instead,
discrimination is generated as a mild, mostly-noise function loosely
tied to the same features, explicitly flagged as a placeholder that
should NOT be treated as literature-grounded the way the difficulty
relationship is.

This covers Y/N Vocab only. ViC needs real sentence content with a
masked word, which is a bigger lift (need a sentence corpus, not just a
word list) -- natural next step, not attempted here.

Usage:
    from simulate_yn_vocab_nlp import simulate_yn_vocab_items_nlp
    items = simulate_yn_vocab_items_nlp(n_items=150, random_seed=42)

Returns the same dict-of-arrays shape as simulate_det.simulate_det_items
(difficulty, discrimination, chance, plus feature columns), but with
richer, named, real feature columns instead of feature_1/feature_2 --
integrating this into autoirt_model.py's FEATURE_COLUMNS is a separate,
follow-up step once this generation logic itself is reviewed.
"""

import numpy as np
from wordfreq import top_n_list, zipf_frequency

MIN_WORD_LEN = 3
MAX_WORD_LEN = 10
CORPUS_SIZE = 20000
NGRAM_SIZES = (2, 3)
TARGET_DIFFICULTY_STD = 1.8  # matches roughly the theta ~ N(0, 2.5) scale used
                              # throughout this project (std ~ 1.58), so items
                              # land in a range abilities actually span


def build_real_word_corpus(corpus_size: int = CORPUS_SIZE) -> list:
    """Real English words, frequency-ranked, filtered to a length range
    that keeps fake-word generation and n-gram tables well-behaved, AND
    intersected with two independently-curated dictionary word lists
    (the 'english-words' PyPI package's web2 and gcide sources) to
    exclude proper nouns and named entities.

    This second filter matters: wordfreq's corpus is built from web
    text, Wikipedia, news, and Reddit, and includes lowercased proper
    nouns and acronyms mixed in with real vocabulary -- checked directly
    and found e.g. "michelle", "nypd", "pakistani", "silva", "warwick",
    "routledge" all present in the top 20,000 words before this filter.
    A vocabulary test item bank should not include those.

    Uses the intersection of two dictionary sources rather than one,
    since either source alone still let some proper-noun-shaped entries
    through (checked directly). The intersection removes 4 of 6 checked
    junk words entirely; the remaining 2 ("pakistani", "silva") are
    defensible edge cases -- both are legitimate dictionary entries in
    their own right (a demonym adjective, and an archaic word for "the
    trees of a region collectively") that happen to also double as a
    name, the same category as "tucker" or "wolverine", which are kept
    correctly. Uses `pip install english-words` (pure Python, data
    bundled in the package, no OS-specific dictionary file, no download
    step) rather than a Linux system dictionary path, so this runs
    identically on Windows, macOS, or Linux."""
    from english_words import get_english_words_set
    dict_words = (get_english_words_set(["web2"], lower=True)
                  & get_english_words_set(["gcide"], lower=True))

    words = top_n_list("en", corpus_size)
    words = [
        w for w in words
        if w.isalpha() and MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN and w in dict_words
    ]
    return sorted(set(words))


def build_ngram_frequency_tables(corpus_words: list, ngram_sizes=NGRAM_SIZES) -> dict:
    """{n: {ngram_string: count}} built from the real corpus, used to
    score wordlikeness for any word or fake word, real or not."""
    tables = {}
    for n in ngram_sizes:
        counts = {}
        for word in corpus_words:
            padded = f"^{word}$"
            for i in range(len(padded) - n + 1):
                gram = padded[i:i + n]
                counts[gram] = counts.get(gram, 0) + 1
        tables[n] = counts
    return tables


def mean_ngram_log_frequency(word: str, ngram_table: dict, n: int) -> float:
    """Average log-frequency of this word's character n-grams against the
    real-corpus table -- the wordlikeness signal. Works for real AND
    fake words, since it only depends on letter patterns, not whether
    the word itself is in the corpus."""
    padded = f"^{word}$"
    grams = [padded[i:i + n] for i in range(len(padded) - n + 1)]
    if not grams:
        return 0.0
    freqs = [np.log1p(ngram_table.get(g, 0)) for g in grams]
    return float(np.mean(freqs))


def build_markov_model(corpus_words: list, order: int = 2):
    """Character Markov model trained on the real corpus, used to
    generate phonotactically plausible non-words. `order` is how many
    preceding characters condition the next-character distribution --
    order=2 (trigram-equivalent: previous 2 chars -> next char) rather
    than order=1 (bigram: previous 1 char -> next char), since order=1
    was checked at scale (300 generated words) and produced a large
    fraction of implausible garbage (e.g. "rve", "wst", "fftivins",
    "rdexeered") -- English phonotactics need at least 2 characters of
    context to meaningfully constrain what can follow. A real
    simplification versus the paper's RNN approach either way, but
    order=2 with the wordlikeness filter in generate_pseudo_words below
    produces markedly more plausible output (verified below)."""
    transitions = {}
    for word in corpus_words:
        padded = "^" * order + word + "$"
        for i in range(len(padded) - order):
            key = padded[i:i + order]
            nxt = padded[i + order]
            transitions.setdefault(key, {})
            transitions[key][nxt] = transitions[key].get(nxt, 0) + 1
    return transitions


def generate_pseudo_words(n_words: int, corpus_words: list, transitions: dict,
                           ngram_tables: dict, random_seed: int, order: int = 2,
                           max_attempts_factor: int = 200,
                           min_wordlikeness_percentile: float = 15.0) -> list:
    """Samples fake words from the Markov model, rejecting anything that
    collapses back into a real word, is outside the length range, OR
    falls below a minimum wordlikeness threshold (its n-gram score must
    clear the `min_wordlikeness_percentile`-th percentile of REAL words'
    own n-gram scores). That last filter matters: even an order-2 Markov
    model still occasionally produces locally-plausible-but-globally-
    garbage strings, and this is the direct, checkable fix for that
    rather than trusting the model alone.

    The real-word exclusion check uses the FULL dictionary (~84k words,
    fetched fresh here), not `corpus_words` (the ~18k frequency-
    restricted set used to train the Markov model and compute
    difficulty). This distinction matters and was found as a real bug
    by inspecting actual output, not assumed: "rind" (a genuine English
    word -- the outer skin of a fruit or cheese) was generated and
    labeled as fake, because it is real but too uncommon to be in the
    frequency-restricted top-20k corpus, so checking only against that
    restricted set missed it. Checking against the full dictionary
    instead catches this regardless of word frequency."""
    from english_words import get_english_words_set
    full_dictionary = (get_english_words_set(["web2"], lower=True)
                        | get_english_words_set(["gcide"], lower=True))
    rng = np.random.default_rng(random_seed)

    real_wordlikeness = np.array([
        (mean_ngram_log_frequency(w, ngram_tables[2], 2)
         + mean_ngram_log_frequency(w, ngram_tables[3], 3)) / 2
        for w in corpus_words
    ])
    min_wordlikeness = float(np.percentile(real_wordlikeness, min_wordlikeness_percentile))

    pseudo_words = set()
    attempts = 0
    max_attempts = n_words * max_attempts_factor

    while len(pseudo_words) < n_words and attempts < max_attempts:
        attempts += 1
        word = ""
        current = "^" * order
        for _ in range(MAX_WORD_LEN + 2):
            options = transitions.get(current)
            if not options:
                break
            chars, counts = zip(*options.items())
            probs = np.array(counts, dtype=float)
            probs /= probs.sum()
            nxt = rng.choice(chars, p=probs)
            if nxt == "$":
                break
            word += nxt
            current = (current + nxt)[-order:]
        if not (MIN_WORD_LEN <= len(word) <= MAX_WORD_LEN) or word in full_dictionary:
            continue
        wordlikeness = (mean_ngram_log_frequency(word, ngram_tables[2], 2)
                         + mean_ngram_log_frequency(word, ngram_tables[3], 3)) / 2
        if wordlikeness >= min_wordlikeness:
            pseudo_words.add(word)

    if len(pseudo_words) < n_words:
        raise RuntimeError(
            f"Only generated {len(pseudo_words)}/{n_words} pseudo-words in {max_attempts} "
            f"attempts -- raise max_attempts_factor or check the Markov model."
        )
    return list(pseudo_words)[:n_words]


def simulate_yn_vocab_items_nlp(n_items: int, real_word_fraction: float = 0.5,
                                 random_seed: int = 42, effect_noise_std: float = 5.5,
                                 discrimination_noise_std: float = 0.1) -> dict:
    """Builds n_items Y/N Vocab items with real content and real
    features. See module docstring for the difficulty/discrimination
    design rationale.

    Returns a dict with:
      "word"                : the actual item content (real or fake word)
      "is_real"              : 1 for real words, 0 for fake
      "length"                : character length
      "zipf_frequency"        : real corpus frequency (0.0 for fake words --
                                 they don't appear in a frequency corpus by
                                 construction, which is itself informative,
                                 not a missing value)
      "mean_bigram_log_freq"  : wordlikeness feature 1 (works for real+fake)
      "mean_trigram_log_freq" : wordlikeness feature 2 (works for real+fake)
      "discrimination", "difficulty", "chance" : true simulated item parameters
    """
    rng = np.random.default_rng(random_seed)
    corpus_words = build_real_word_corpus()
    ngram_tables = build_ngram_frequency_tables(corpus_words)
    transitions = build_markov_model(corpus_words, order=2)

    n_real = int(round(n_items * real_word_fraction))
    n_fake = n_items - n_real

    real_words = list(rng.choice(corpus_words, size=n_real, replace=False))
    fake_words = generate_pseudo_words(n_fake, corpus_words, transitions, ngram_tables, random_seed + 1)

    words = real_words + fake_words
    is_real = np.array([1] * n_real + [0] * n_fake)

    length = np.array([len(w) for w in words], dtype=float)
    zipf_freq = np.array([
        zipf_frequency(w, "en") if real else 0.0
        for w, real in zip(words, is_real)
    ])
    bigram_feat = np.array([mean_ngram_log_frequency(w, ngram_tables[2], 2) for w in words])
    trigram_feat = np.array([mean_ngram_log_frequency(w, ngram_tables[3], 3) for w in words])

    # --- Difficulty: literature-grounded, asymmetric by is_real ---
    # Real words: harder as frequency drops (standardize zipf among real
    # words only, since 0.0 for fakes isn't a meaningful point on this scale).
    real_zipf_mean = zipf_freq[is_real == 1].mean()
    real_zipf_std = zipf_freq[is_real == 1].std()
    frequency_effect = np.where(
        is_real == 1,
        -(zipf_freq - real_zipf_mean) / (real_zipf_std + 1e-8),  # rarer real word -> higher difficulty
        0.0,
    )
    # Fake words: harder as wordlikeness rises (standardize among fakes only).
    wordlikeness = (bigram_feat + trigram_feat) / 2
    fake_wordlikeness_mean = wordlikeness[is_real == 0].mean()
    fake_wordlikeness_std = wordlikeness[is_real == 0].std()
    wordlikeness_effect = np.where(
        is_real == 0,
        (wordlikeness - fake_wordlikeness_mean) / (fake_wordlikeness_std + 1e-8),
        0.0,
    )
    difficulty_noise = rng.normal(0, effect_noise_std, size=n_items)
    difficulty_raw = 1.5 * (frequency_effect + wordlikeness_effect) + difficulty_noise
    # Rescale to a sensible absolute range relative to the ability
    # distribution (theta ~ N(0, 2.5), std ~ 1.6 elsewhere in this
    # project). The noise level above was deliberately calibrated to
    # match a realistic literature correlation magnitude (see docstring),
    # which as a side effect inflates the ABSOLUTE spread of difficulty
    # far past where it should sit -- an item with difficulty +/-15 is
    # unanswerable-or-trivial for essentially every realistic test-taker,
    # since the logistic curve saturates within a few units of
    # |theta - difficulty|. Rescaling by a constant preserves the
    # correlation exactly (correlation is scale-invariant) while fixing
    # the absolute range to TARGET_DIFFICULTY_STD, comparable to the old
    # synthetic setup's difficulty spread.
    difficulty = difficulty_raw * (TARGET_DIFFICULTY_STD / difficulty_raw.std())

    # --- Discrimination: weak, mostly noise, NOT literature-grounded --
    # see module docstring. Loosely: items with more extreme (very easy
    # or very hard) generating signal get a small discrimination bump,
    # a common but contested finding, kept deliberately mild here.
    # Uses its OWN, independently-calibrated noise parameter, not
    # effect_noise_std -- discrimination lives on a multiplicative
    # (log-space, exp()'d) scale with a narrow realistic range (the old
    # synthetic setup's true discrimination was roughly 0.5-1.5), so
    # reusing difficulty's much larger additive noise scale here would
    # blow that range out to something degenerate.
    extremity = np.abs(frequency_effect + wordlikeness_effect)
    discrimination_signal = 0.05 * extremity
    discrimination_noise = rng.normal(0, discrimination_noise_std, size=n_items)
    discrimination = np.exp(np.log(1.0) + discrimination_signal + discrimination_noise)

    chance = np.full(n_items, 0.25)  # matches YN_CHANCE, multiple-choice guessing

    return {
        "word": np.array(words),
        "is_real": is_real,
        "length": length,
        "zipf_frequency": zipf_freq,
        "mean_bigram_log_freq": bigram_feat,
        "mean_trigram_log_freq": trigram_feat,
        "discrimination": discrimination,
        "difficulty": difficulty,
        "chance": chance,
    }


if __name__ == "__main__":
    # Smoke test / sanity check when run directly.
    items = simulate_yn_vocab_items_nlp(n_items=150, random_seed=42)
    print(f"Generated {len(items['word'])} items "
          f"({items['is_real'].sum()} real, {(1 - items['is_real']).sum()} fake)")
    print("\nSample real words:", items["word"][items["is_real"] == 1][:8].tolist())
    print("Sample fake words:", items["word"][items["is_real"] == 0][:8].tolist())
    print(f"\nDifficulty: mean={items['difficulty'].mean():.3f}, std={items['difficulty'].std():.3f}, "
          f"range=[{items['difficulty'].min():.3f}, {items['difficulty'].max():.3f}]")
    print(f"Discrimination: mean={items['discrimination'].mean():.3f}, std={items['discrimination'].std():.3f}, "
          f"range=[{items['discrimination'].min():.3f}, {items['discrimination'].max():.3f}]")

    import numpy as _np
    real_mask = items["is_real"] == 1
    r_real = _np.corrcoef(items['zipf_frequency'][real_mask], items['difficulty'][real_mask])[0, 1]
    print(f"\nSanity check -- within real words, correlation(zipf_frequency, difficulty): "
          f"{r_real:.3f} (should be negative, and roughly literature-scale, r~0.15-0.35, "
          f"not near -1: Ha et al. 2019's best COMBINED feature model only reached r=0.32)")
    fake_mask = items["is_real"] == 0
    wordlikeness_fake = (items["mean_bigram_log_freq"][fake_mask] + items["mean_trigram_log_freq"][fake_mask]) / 2
    r_fake = _np.corrcoef(wordlikeness_fake, items['difficulty'][fake_mask])[0, 1]
    print(f"Sanity check -- within fake words, correlation(wordlikeness, difficulty): "
          f"{r_fake:.3f} (should be positive, similar realistic scale)")
    print(f"\nDiscrimination range check: should stay roughly in [0.5, 1.5], matching the old "
          f"synthetic setup's true range -- min={items['discrimination'].min():.3f}, "
          f"max={items['discrimination'].max():.3f}")