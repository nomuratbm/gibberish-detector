"""Rule-based pre/post filters layered on top of the ML classifier.

Rationale. The DistilBERT classifier has two consistent failure modes that
matter for real user content:

1. Repetitive sub-word noise such as "awawawaawawawaawawawa" is tokenized
   into valid-looking sub-pieces and labeled `mild_giberish` instead of
   `noise`. When the deployment's `GIBBERISH_LABELS` excludes mild_gibberish
   this becomes a false negative.
2. Short, title-cased, non-sentence strings such as "Tech Week 2026" also
   come back as `mild_gibberish` because they lack sentence structure. When
   mild_gibberish IS flagged this becomes a false positive.

We address both with three cheap, deterministic rules that run alongside the
model call:

* Rule 1 (force gibberish). The string has enough alphabetic content to judge
   (>= FORCE_GIBBERISH_MIN_ALPHA letters), is NOT short, and contains no
   dictionary word of at least MIN_REAL_WORD_LENGTH characters. This catches
   "awawawa..." regardless of what the model says.
* Rule 2 (force clean). The string IS short AND contains a real dictionary
   word. This lets event titles, product names and dates pass even if the
   model would have called them mild_gibberish.
* Rule 3 (confidence gate). If the model picks a gibberish label but its
   softmax score is below CONFIDENCE_THRESHOLD, downgrade to not-gibberish.
   Only applies to model output.

All three rules are individually disable-able:
* Rule 1 and Rule 2 are skipped entirely when ENABLE_HEURISTICS=false.
* Rule 1 is disabled by setting FORCE_GIBBERISH_MIN_ALPHA=0.
* Rule 2 is disabled by setting SHORT_INPUT_MAX_CHARS=0.
* Rule 3 is disabled by setting CONFIDENCE_THRESHOLD=0.

`_COMMON_WORDS` is a hand-maintained set of very frequent English words. It
is intentionally compact; add domain-specific entries (event names, product
codes, etc.) below if your site keeps tripping over them.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split on any run of non-alphanumeric characters and lower-case.

    Digits survive as their own tokens, which lets the word-set check ignore
    numeric fragments like "2026" without treating them as words.
    """
    return [tok for tok in _NON_ALNUM.split(text.lower()) if tok]


def alpha_letter_count(text: str) -> int:
    """Count alphabetic characters only, ignoring digits, symbols, spaces."""
    return sum(1 for ch in text if ch.isalpha())


# ---------------------------------------------------------------------------
# Rule predicates (each returns True/False; the caller combines with config)
# ---------------------------------------------------------------------------


def has_real_word(text: str, min_length: int) -> bool:
    """True if any token of at least `min_length` chars is in the word set."""
    return any(
        len(tok) >= min_length and tok in _COMMON_WORDS for tok in tokenize(text)
    )


def is_short_input(text: str, max_chars: int, max_tokens: int) -> bool:
    """A string is 'short' only if BOTH limits are met.

    Using AND here matters: a long single-token string like
    "awawawaawawawaawawawa" has only one token, so an OR test would
    classify it as short and prevent Rule 1 from firing. Requiring both
    limits to hold correctly treats it as long garbage.

    Any zero or negative limit disables the whole check (nothing is
    short), which is also how Rule 2 is turned off via env.
    """
    if max_chars <= 0 or max_tokens <= 0:
        return False
    stripped = text.strip()
    return len(stripped) <= max_chars and len(tokenize(stripped)) <= max_tokens


def should_force_gibberish(
    text: str,
    min_word_len: int,
    min_alpha: int,
    short_max_chars: int,
    short_max_tokens: int,
) -> bool:
    """Rule 1: long, alphabetic, no dictionary word of >= min_word_len."""
    if min_alpha <= 0:
        return False
    if alpha_letter_count(text) < min_alpha:
        return False
    if is_short_input(text, short_max_chars, short_max_tokens):
        return False
    return not has_real_word(text, min_word_len)


def should_force_clean(
    text: str,
    min_word_len: int,
    short_max_chars: int,
    short_max_tokens: int,
) -> bool:
    """Rule 2: short AND has at least one real dictionary word."""
    if short_max_chars <= 0 and short_max_tokens <= 0:
        return False
    if not is_short_input(text, short_max_chars, short_max_tokens):
        return False
    return has_real_word(text, min_word_len)


# ---------------------------------------------------------------------------
# Compact common-English word set.
#
# Only entries of length 4+ are actually consulted at runtime, because the
# caller enforces MIN_REAL_WORD_LENGTH, but shorter duplicates are still
# allowed here for future tuning. All are plain dictionary vocabulary; feel
# free to add project-specific nouns (brand names, event names, product
# codes) without touching any other file.
# ---------------------------------------------------------------------------

_COMMON_WORDS: frozenset[str] = frozenset(
    """
    able about above across act action active activity add address after
    again against age agent ago agree air all allow almost alone along
    already also although always am amount an and animal another answer
    any anyone anything area arm around arrive art as ask at attention
    available average away back bad bag ball bank bar base be beat beauty
    became because become bed been before began begin behavior behind
    being believe bell below beside best better between beyond big bird
    bit black blue board boat body book born both box boy break bring
    brother build building business but buy by call came can cannot car
    card care carry case catch cat cause center certain chair chance
    change check child choice choose church city class clear close
    cold college color come common community company complete computer
    condition consider contain continue control cook cool copy corner
    cost could country course cover create cross cry cup current cut
    dark data day dead deal death decide decision deep design detail
    development did die difference different difficult direct direction
    discover discussion do dog door double draw dream dress drink drive
    drop during each ear early easy eat economic edge effect effort egg
    eight either else end energy enjoy enough enter entire equal
    especially even event ever every everyone everything except exercise
    expect experience explain eye face fact family far fast father feel
    field fifteen fifty fight fill film final find fine finger finish
    fire first fish five floor flow flower fly follow food foot for
    force foreign form forward four free friend from front full fun
    game garden gas general get girl give glass go gold good government
    green ground group grow guess guide gun had hair half hand happen
    hard head hear heart heavy held help her here high hill him himself
    his history hit hold home hope hospital hot hour house how however
    huge human hundred hurt husband idea if imagine important improve
    in inch include indeed information inside instead interest into
    iron island it its itself job join journal journey judge jump just
    keep key kid kill kind kitchen knew know knowledge known
    lack land language large last late laugh law lead learn least leave
    led left leg less let letter level lie life light like line list
    listen little live long look lose loss lost lot love low machine
    main major make man many map mark market marry match matter may
    maybe me mean measure meat media meet member memory men mention
    message method mid middle might mile milk mind minute miss modern
    moment money month more morning most mother mouth move movie much
    music must my name nation natural nature near necessary need never
    new next nice night nine no none nor north not note nothing notice
    now number nuclear occur off offer office often oil old on once one
    only open operation option or order other our out outside over own
    page pair paper parent part particular party pass past pattern pay
    people per perhaps period person pick picture piece place plan play
    please point police policy poor position possible power prepare
    present president press pretty price print probably problem process
    produce product program project provide public pull purpose push
    put question quick quiet quite race radio raise range rather reach
    read ready real reason receive recent record red regard region
    remain remember report require research resource respond result
    return reveal rich ride right ring rise risk river road rock role
    room rise run sad safe said same sat save saw say scene school
    science sea search season second section see seed seem seen self
    sell send sense series serious serve service set seven several
    shadow shall shape share she short should shot sick side sign
    simple since sing single sister sit site six size skill sleep small
    smart smile snow so social some something sometimes son song soon
    sort sound source south space speak special speed spend spirit
    spot spring staff stage stand star start state station stay steel
    step still stock stone stop store story street strong student study
    stuff subject success such sudden suggest summer sun support sure
    surface system table take talk task tax teach team tell ten term
    test than thank that the their them then there these they thick
    thing think third this those though thought thousand three through
    throw thus time tiny to today together tomorrow tone tonight too
    took tool top topic total touch toward town trade training travel
    tree trip trouble true trust try turn two type under understand
    unit until up upon us use useful usually value various very view
    visit voice wait walk wall want war watch water way we week weight
    welcome well went were what when where whether which while who
    whole whom whose why wide wife will win wind window wish with within
    without woman wonder word work world worry worth would write writer
    wrong year yes yet you young your youth
    tech app api website online digital event summit festival launch
    version release update design developer software hardware network
    security privacy cloud server email message project research
    """.split()
)
