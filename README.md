# Harmonizer

Harmonizer takes a monophonic melody and predicts a chord progression to accompany it, trained on the Nottingham folk dataset. The core is a bidirectional LSTM that reads pitch and duration as parallel input sequences and outputs a chord label at every timestep, framing harmonization as sequence labeling rather than free generation. Beyond the standard accuracy numbers, evaluation includes a custom functional accuracy that estimates each song's key and checks whether the model's mistakes still make sense harmonically, rather than treating every wrong answer as equally wrong.

## Results

### Model comparison
*(per-note argmax decoding, val split)*

| Model | Micro | Macro | Root | Functional |
|---|---|---|---|---|
| Majority class (G:maj) | 0.2499 | — | — | — |
| BiLSTM (2.57M params) | 0.6497 | 0.4042 | 0.7239 | 0.7731 |

### Decoding ablation (BiLSTM, val split)

| Decoder | Micro | Macro | Root | Functional | Changes/song |
|---|---|---|---|---|---|
| Per-note argmax | 0.6497 | 0.4042 | 0.7239 | 0.7731 | 42.54 |
| Viterbi, uniform penalty 0.25 | 0.6505 | 0.3974 | 0.7237 | 0.7723 | 35.54 |
| Viterbi, learned transitions λ=0.05 | 0.6507 | 0.4005 | 0.7237 | 0.7729 | 36.92 |
| *Ground truth* | | | | | *37.47* |

Accuracy barely moves across decoders, but change rate does: per-note
argmax over-changes by about 13% relative to the ground truth's 37.47,
and both Viterbi variants pull that down closer to true. See Findings
for what's actually driving the difference.

- **Micro accuracy** — exact chord match, every note weighted equally.
- **Macro accuracy** — exact chord match, averaged per chord class rather than per note, so rare chord types count as much as common ones.
- **Root accuracy** — credit given if the predicted chord's root matches, regardless of quality (e.g. predicting C:maj when the answer was C:dom7 still counts).
- **Functional accuracy** — credit given if the predicted and true chords share the same harmonic function (tonic/predominant/dominant) relative to the song's estimated key, since multiple chords can serve the same musical role.

## Data 

The model is trained on the [Nottingham Music Database](https://github.com/jukedeck/nottingham-dataset), a collection 
of ~1000 British/Irish folk melodies with aligned chord annotations, 
originally in ABC notation and converted to MIDI.

Songs are split 80/10/10 into train/val/test **before** any augmentation, 
split by song ID so no melody (or a transposed variant of it) appears in 
more than one split. Only the training split is augmented. Each training 
song is transposed into all 12 keys (semitone shifts of -5 to +6), giving 
the model roughly 12x more training sequences and teaching it key 
invariance directly from data. Validation and test songs are left in their 
original keys, so reported metrics reflect performance on real, unmodified melodies. 
13 of 1,034 files were rejected during parsing as melody-only transcriptions with no accompaniment track.

| Split | Songs | Notes | Pitch range |
|-------|-------|-------|-------------|
| Train (post-transposition) | 816 songs × 12 keys = 9,792 sequences | 1,870,332 | 50–94 |
| Val | 102 | 18,034 | 55–86 |
| Test | 103 | 19,053 | 57–84 |

Chord labels are represented as root + quality (e.g. `C:maj`, `G:dom7`) 
against a generated 12-root × 10-quality vocabulary (121 classes + `<PAD>`), 
rather than a vocabulary collected from the raw data — see 
[Tokenization Design](#tokenization--data-design) for the full reasoning.

## Tokenization & Data Design

### Output vocabulary: chord labels
**Decision:** Define the output vocabulary as the full grid of 12 roots x 
10 quality classes observed in the data (maj, min, dim, aug, dom7, min7, 
maj7, hdim7, dim7, oth), plus a no-chord label `NC` and `<PAD>` (a special token in every vocab because every song has different lengths, but a training batch is a rectangular tensor): **120 cells, 121 chord classes, and 122 output units total**. 
This is used instead of either the 54 root-quality combinations literally 
observed in the raw data or a reduced common-quality label set.

**Why:** The label set must be closed under transposition, because the 
training pipeline transposes every song into all 12 keys (see below). 
Transposing a song containing G#:aug produces aug chords on all 12 roots, 
most of which never appear in the untransposed data — so "the 54 observed 
labels" stops being a valid vocabulary the moment augmentation runs. The 
full root x quality grid is closed under transposition by construction. It 
also preserves every harmonic distinction present in the source material, 
and augmentation gives each rare quality class 12x the examples, spread 
across all roots — so the long tail gets meaningfully more training signal 
than it would in a single-key setup.

**Considered instead:** (a) Keeping only the 54 observed labels — 
contradicts transposition augmentation, as above. (b) Collapsing to a 
reduced quality set (e.g. maj/min/dom7 + other) — simpler, with better 
per-class accuracy, but erases real chords present in the source material.

**Tradeoff:** Many of the 122 output units remain rare even after 
augmentation, and the model will likely underperform or never predict 
them. To account for this honestly, results are reported as both micro 
accuracy (overall, dominated by common classes) and macro accuracy 
(averaged per class, exposing long-tail performance) rather than a single 
headline number.

### Melody input representation: key transposition
**Decision:** Augment the dataset by transposing every song into all 12 
keys, and retain raw MIDI pitch (including octave) rather than abstracting 
to pitch-class only. Melody events are encoded as two parallel inputs to keep the vocabulary smaller. 

**Why:** Transposition augmentation teaches the model key invariance 
directly from data (12x more training sequences), which is a cleaner path 
to invariance than throwing away octave information architecturally. Octave 
is preserved for melody input since it's real information the model can 
use; it's dropped from chord *output* since chord labels here are voicing-
free by design.

**Considered instead:** Pitch-class-only melody encoding (also achieves key 
invariance, but loses octave/register information that may correlate with 
chord choice).

**Tradeoff:** Larger effective dataset size and longer training time. Also 
sets up straightforward extension to other MIDI datasets/genres later, since 
the transposition pipeline is genre-agnostic.

**Implementation note:** Train/val/test split is performed on original song 
IDs *before* transposition, then the training split is transposed independently. 
Splitting after transposition would leak near-duplicate transposed versions 
of the same song across splits and inflate validation accuracy.

### Duration handling
**Decision:** Represent duration as a categorical token per distinct 
observed value (e.g. 0.5, 1.0, 1.5, including triplet values), rather than 
quantizing to a fixed grid. Duration values occurring fewer than 10 times 
in the training set are folded into a single `<RARE_DUR>` bucket token.

**Why:** A fixed 16th-note grid would misrepresent triplet rhythms present 
in the dataset by forcing them onto grid positions that don't reflect their 
actual timing. Categorical tokens preserve exact rhythm values as they 
appear in the source data. The rare-value fallback keeps the vocabulary 
from being polluted by one-off/likely-erroneous duration values while 
still allowing any duration to be handled gracefully at inference time.

**Considered instead:** Fixed-grid quantization (simpler, standard in some 
literature, but destroys triplet feel); continuous/regression-based 
duration (more precise, but a worse fit for an otherwise categorical 
pipeline).

**Tradeoff:** Categorical duration tokens carry no inherent ordinality — 
the model isn't told that 1.5 sits "between" 1.0 and 2.0, and can only 
learn that relationship from co-occurrence patterns if there's enough data. 
This is a known limitation of the current scheme, not something the 
`<RARE_DUR>` bucket solves. At inference, any out-of-vocabulary duration 
value falls back to `<RARE_DUR>` rather than erroring.

### Genre Token

**Decision:** Add a genre token for future dataset expansion purposes.

**Why:** The current dataset being considered is the Nottingham Music
Dataset, which includes only folk song tunes. Adding future datasets is
possible, but mixing between different genres like pop and folk leads to
the model learning from contradictory styles and producing worse results.
`<folk>` is reserved in the tokenizer vocabulary for future multi-genre
conditioning and is not yet fed to the model.

## Training Setup

The model is a bidirectional LSTM: 2 layers, hidden dim 256, with separate
embedding tables for pitch (dim 128) and duration (dim 64) that are
concatenated at each timestep before entering the LSTM. Dropout of 0.3 is
applied for regularization.

Trained with Adam (lr=1e-3) and cross-entropy loss, masked to ignore
`<PAD>` positions so padded timesteps don't contribute to the gradient.
Batch size 32, for 20 epochs, with the checkpoint saved on best validation
loss rather than final epoch (`artifacts/model.pt`). Training and
validation loss, plus token-level validation accuracy, are logged to
Weights & Biases each epoch.

| Hyperparameter | Value |
|---|---|
| Hidden dim | 256 |
| LSTM layers | 2 |
| Pitch embedding dim | 128 |
| Duration embedding dim | 64 |
| Dropout | 0.3 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 32 |
| Epochs | 20 |
| Loss | Cross-entropy (PAD-masked) |
| Checkpoint selection | Best validation loss |

## Findings

The baseline BiLSTM predicts chords independently at each note, which
produces harmony that rapidly flickers. The output still makes musical
sense note-to-note, but these are patterns that do not appear in folk
harmony.

A uniform switch penalty fixes the flicker but sounded worse by ear. The
model's per-note scores already account for the melody, so the penalty
isn't blind to pitch clashes; it's blind to *which* change is happening,
since every change costs the same regardless of how common it is. The
clearest case: idiomatic V–I back-and-forth got flattened into just
staying on the tonic, because switching away and back costs twice, no
matter how natural that motion is harmonically.

A transition-matrix Viterbi (a 122×122 log-probability table of
chord-to-chord transitions, Laplace-smoothed and estimated from the
training split) replaced the flat penalty with a learned prior over which
changes are actually common. At λ=0.05 this narrowed the predicted/true
change-rate gap and sounded modestly better than the uniform-penalty
version, but still overrode some valid musical choices, since the matrix
only knows chord identity, not timing or the note being harmonized.

There were three other findings from this pass that are independent of decoding:

- **Overfitting.** Validation loss bottomed out at epoch 11 while training
  loss kept falling; the checkpoint logic correctly kept the epoch-11
  model rather than the final one.
- **Padding contamination.** Batched validation (with padding) measured
  0.6568 accuracy; the clean, unpadded measurement used for the results
  table above is 0.6497. The gap comes from the backward LSTM reading
  padding tokens before it reaches the real notes.
- **Functional vs. exact accuracy.** Exact match is 65%, functional match
  is 77% — roughly a third of the model's "wrong" answers are chords that
  serve the same harmonic role as the true answer, not random misses.

## How To Run

```bash
git clone --depth 1 https://github.com/jukedeck/nottingham-dataset data/nottingham
uv run python scripts/parse.py
uv run python scripts/split.py
uv run wandb login
uv run python train.py
uv run python transition.py
uv run python eval.py
uv run python render.py
```

## Roadmap

- **Beat-aware penalty**: incorporate note offset/duration into the decode
  step so chord-change cost depends on metrical position, not just chord
  identity.
- **Jointly-trained CRF**: replace the two-stage LSTM-then-Viterbi pipeline
  with a linear-chain CRF trained end-to-end, so transition scores are
  learned jointly with the emission model rather than estimated separately
  post-hoc.
- **Transformer comparison**: swap the BiLSTM for a Transformer encoder and
  compare against the current results table.
- **One-time test-set evaluation**: run the held-out test split once,
  after model/decoder selection is finalized on val.
- **Gradio demo**: a simple interface to upload or play a melody and hear
  the predicted harmonization.
- **Genre conditioning**: train on POP909 alongside Nottingham, using the
  `<folk>`/`<pop>` genre token that's currently reserved but unused.