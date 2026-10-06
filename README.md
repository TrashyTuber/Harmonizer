# Harmonizer

**[Try the live demo →](https://huggingface.co/spaces/Trazhytuber/Melody_Harmonizer)**

Type or upload a melody and hear it harmonized.

Harmonizer takes a monophonic melody and predicts a chord progression to accompany it, trained on the Nottingham folk dataset. The core is a bidirectional LSTM that reads pitch and duration as parallel input sequences and outputs a chord label at every timestep, framing harmonization as sequence labeling rather than free generation. A size-matched Transformer encoder trained under the same setup scored about 5 points lower across three runs each, and 6 points lower on the held-out test set. Beyond the standard accuracy numbers, evaluation includes a custom functional accuracy that estimates each song's key and checks whether the model's mistakes still make sense harmonically, rather than treating every wrong answer as equally wrong.

## Results

### Model comparison
*(per-note argmax decoding, val split)*

| Model | Micro | Macro | Root | Functional |
|---|---|---|---|---|
| Majority class (G:maj) | 0.2499 | — | — | — |
| BiLSTM (2.57M params) | 0.634 ± 0.010 | 0.372 ± 0.021 | 0.700 ± 0.012 | 0.748 ± 0.010 |
| Transformer (2.46M params) | 0.584 ± 0.002 | 0.332 ± 0.008 | 0.645 ± 0.001 | 0.695 ± 0.002 |

*3 training runs per model (2 seeded, 1 unseeded)*

### Decoding ablation (original single-run BiLSTM, `artifacts/model.pt`, val split)

| Decoder | Micro | Macro | Root | Functional | Changes/song |
|---|---|---|---|---|---|
| Per-note argmax | 0.6497 | 0.4042 | 0.7239 | 0.7731 | 42.54 |
| Viterbi, uniform penalty 0.25 | 0.6505 | 0.3974 | 0.7237 | 0.7723 | 35.54 |
| Viterbi, learned transitions λ=0.05 | 0.6507 | 0.4005 | 0.7237 | 0.7729 | 36.92 |
| *Ground truth* | | | | | *37.47* |

This is the original BiLSTM run (20 epochs, whole-song training), so its
argmax row (0.6497) differs from the 3-run mean in the model comparison
table above.

Accuracy barely moves across decoders, but change rate does: per-note
argmax over-changes by about 13% relative to the ground truth's 37.47,
and both Viterbi variants pull that down closer to true. See Findings
for what's actually driving the difference.

- **Micro accuracy** — exact chord match, every note weighted equally.
- **Macro accuracy** — exact chord match, averaged per chord class rather than per note, so rare chord types count as much as common ones.
- **Root accuracy** — credit given if the predicted chord's root matches, regardless of quality (e.g. predicting C:maj when the answer was C:dom7 still counts).
- **Functional accuracy** — credit given if the predicted and true chords share the same harmonic function (tonic/predominant/dominant) relative to the song's estimated key, since multiple chords can serve the same musical role.

## Demo

The model is deployed as a [Hugging Face Space](https://huggingface.co/spaces/Trazhytuber/Melody_Harmonizer) running on CPU hardware. A 2.5M-parameter model needs no GPU.

- **Input and output.** Enter a melody as text notation or upload a MIDI file. You get back audio, a chord chart, and a MIDI download. A Raw/Smoothed switch lets you hear what the decoder changes: Raw is per-note argmax, Smoothed applies the learned-transition Viterbi decoder.
- **Any key works.** There is no transposition step at inference, because training covered all 12 keys. This is a direct payoff of the augmentation decision above.
- **Out-of-range input.** Melodies outside the trained pitch range are shifted by octaves before the model sees them. Unusual note lengths map to `<RARE_DUR>`.
- **Example tunes** are public-domain melodies, not dataset songs, so they show the model handling music it has never seen. This also means the Space doesn't redistribute the GPL-licensed dataset.
- **Known limitation.** The model predicts N.C. for the opening notes of most melodies, because it has no beat-position input to tell a pickup from a downbeat start. The demo fills these notes with the first predicted chord for now.

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

**Tradeoff:** Many of the 121 chord classes remain rare even after 
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

Both models take the same inputs (pitch and duration embeddings,
concatenated at each note) and output chord logits per note, so the
comparison changes only the encoder.

**BiLSTM.** 2 bidirectional layers with hidden dim 256, and dropout 0.3
applied between the LSTM layers.

**Transformer.** The embeddings are projected to d_model 256 and passed
through a 3-layer encoder with 4 attention heads, feedforward dim 1024,
dropout 0.1, and pre-norm layers. Attention is unmasked, so each note sees
the whole melody in both directions, and padded positions are ignored
through a padding mask. Positions use fixed sinusoidal encodings instead
of a learned table, so any song length works (the longest songs in the
dataset reach about 2,700 notes).

Both models train with Adam (lr 1e-3) and cross-entropy loss that ignores
`<PAD>` positions, batch size 32, for 30 epochs. Training uses 512-note
windows, which keeps the transformer's attention memory manageable.
Validation uses whole songs. The checkpoint with the best validation loss
is kept (`artifacts/lstm.pt` and `artifacts/transformer.pt`), with the
training config stored inside so `eval.py` can rebuild the model. Each
model was trained 3 times (one unseeded, seeds 42 and 57), and the results
tables report the mean ± std. Loss and token-level validation accuracy are
logged to Weights & Biases each epoch.

| | BiLSTM | Transformer |
|---|---|---|
| Layers | 2 | 3 |
| Hidden size / d_model | 256 | 256 |
| Attention heads | n/a | 4 |
| Feedforward dim | n/a | 1024 |
| Pitch / duration embedding dim | 128 / 64 | 128 / 64 |
| Dropout | 0.3 (between layers) | 0.1 |
| Position information | recurrence | sinusoidal encoding |
| Optimizer / learning rate | Adam / 1e-3 | Adam / 1e-3 |
| Batch size | 32 | 32 |
| Epochs | 30 | 30 |
| Training input | 512-note windows | 512-note windows |
| Validation input | whole songs | whole songs |
| Checkpoint selection | best val loss | best val loss |

## Findings

The baseline BiLSTM decodes each note independently, which
produces harmony that rapidly flickers. The output still makes musical
sense note to note, but these are patterns that do not appear in folk
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
only knows chord identity, not timing or the note being harmonized. The
decoding ablation comes from a single BiLSTM run, so its accuracy
differences (about 0.001) are far inside the run-to-run spread described
below. The change rate is the measure that actually moves.

Other findings from this pass, independent of decoding:

- **BiLSTM vs. Transformer.** Averaged over three training runs each, the
  BiLSTM beats the Transformer on every metric, by 4 to 5.5 points (micro
  0.634 vs. 0.584, functional 0.748 vs. 0.695). The Transformer also
  changes chords more often (54.2 per song vs. 42.1, against a true rate
  of 37.5), so its output flickers more than the BiLSTM's. A likely reason
  is that the LSTM's sequential memory suits this task, and 816 songs is
  not enough for the Transformer to learn note-to-note smoothness on its
  own. This explanation has not been tested directly.
- **Run-to-run variation.** Across the three BiLSTM runs, micro accuracy
  ranged from 0.623 to 0.644, so comparing single runs can mislead. (The
  original run, trained for 20 epochs without windowing, reached 0.650.)
  The Transformer was much more stable (std 0.002 vs. 0.010).
- **Overfitting.** In the original BiLSTM run (20 epochs, whole songs),
  validation loss bottomed out at epoch 11 while training loss kept
  falling; the checkpoint logic correctly kept the epoch-11 model rather
  than the final one.
- **Padding contamination.** For that same original checkpoint
  (`artifacts/model.pt`), batched validation (with padding) measured
  0.6568 accuracy, versus 0.6497 for the clean, unpadded evaluation used
  in the decoding ablation. The gap comes from the backward LSTM reading
  padding tokens before it reaches the real notes. This does not apply to
  the Transformer, whose attention ignores padded positions through a
  padding mask.
- **Functional vs. exact accuracy.** For the BiLSTM, exact match averages
  63% and functional match 75%, so roughly a third of its "wrong" answers
  are chords that serve the same harmonic role as the true answer, not
  random misses.
- **Opening N.C.** 78 of 102 real validation songs open with N.C. (no chord), and the model does so on 84. The model has learned the habit of starting with no chords from the data, but it has no beat-position input, so it can't check whether a song's opening notes are actually a pickup or the first full bar. 
- **Repeated phrases.** Repeated phrases are measured on the 92 val songs with exact melodic repeats of 16+ notes, one checkpoint per model; the real tunes score 0.954. The BiLSTM gave repeated phrases the same chords more often than the Transformer (0.880 vs. 0.836). This was unexpected, since attention can in principle compare distant phrases directly. One untested explanation is that the Transformer's higher chord-change rate (54.2 vs. 42.1 per song) makes its output less consistent overall.

## Test Set Evaluation

The test split (103 songs) was held out through all model and decoder
selection. It is evaluated once, with the choices below, which were fixed
before any test result was seen.

**Final choices (fixed before evaluation):**
- Model: BiLSTM, seed-57 checkpoint (selected for best validation micro
  accuracy among the three BiLSTM runs)
- Decoder: learned transitions, λ = 0.05
- Comparison point: Transformer, seed-57 checkpoint, evaluated with both decoders

| Model | Decoder | Micro | Macro | Root | Functional | Changes/song |
|---|---|---|---|---|---|---|
| BiLSTM | Argmax | 0.6671 | 0.3664 | 0.7204 | 0.7664 | 46.9 |
| BiLSTM | Learned transitions, λ=0.05 | 0.6681 | 0.3622 | 0.7217 | 0.7676 | 41.7 |
| Transformer | Argmax | 0.6069 | 0.2928 | 0.6535 | 0.7048 | 61.4 |
| Transformer | Learned transitions, λ=0.05 | 0.6133 | 0.2964 | 0.6597 | 0.7103 | 50.4|
| *Ground truth* | | | | | | 43.5 |

The differences in numbers compared to the validation tables are to be expected due to evaluation coming from a different set of about 100 songs. The transition decoder changes chords slightly less than the ground truth now as λ was tuned on val, where the true change rate was 37.5 per song, against 43.5 on test.

## How To Run

```bash
git clone --depth 1 https://github.com/jukedeck/nottingham-dataset data/nottingham
uv run python scripts/parse.py
uv run python scripts/split.py
uv run wandb login
uv run python train.py
```

The architecture is set by `"arch"` in the config at the top of
`train.py` (`"lstm"` or `"transformer"`). Run `train.py` once per
architecture to produce both checkpoints.

```bash
uv run python transition.py
uv run python eval.py
uv run python eval.py --checkpoint artifacts/transformer.pt
uv run python render.py

# Local demo (needs FluidSynth: `brew install fluid-synth` on macOS)
uv run python app.py

# Deploy to the Hugging Face Space
uv run python scripts/build_space.py
uv run hf upload Trazhytuber/Melody_Harmonizer space_build . --repo-type space
```

## Project Structure

- `scripts/parse.py`, `scripts/split.py`: MIDI parsing and the song-level train/val/test split
- `scripts/build_space.py`: packages the demo for Hugging Face
- `tokenizer.py`, `dataset.py`: vocabularies, encoding, batching
- `model.py`: BiLSTM and Transformer, plus the `build_model` factory
- `train.py`: training (architecture chosen by `"arch"` in its config)
- `transition.py`, `decode.py`: transition matrix and Viterbi decoders
- `eval.py`: metrics (micro, macro, root, functional, change rate)
- `render.py`: writes predicted and true chords to MIDI for listening
- `checkpoint.py`: rebuilds a model from a checkpoint
- `app.py`: the Gradio demo
- `space/`: the Space's config files

## License

Code is licensed under GPL-3.0. The model is trained on the
[Nottingham Music Database](https://github.com/jukedeck/nottingham-dataset),
also GPL-3.0.

## Roadmap

- **Beat-position input**: give the model each note's position in the bar.
  This is the likely fix for the opening N.C. and the early chord entries,
  and it is also what the beat-aware penalty below needs.
- **Beat-aware penalty**: incorporate note offset/duration into the decode
  step so chord-change cost depends on metrical position, not just chord
  identity.
- **Jointly-trained CRF**: replace the two-stage LSTM-then-Viterbi pipeline
  with a linear-chain CRF trained end-to-end, so transition scores are
  learned jointly with the emission model rather than estimated separately
  post-hoc.
- **Genre conditioning**: train on POP909 alongside Nottingham, using the
  `<folk>`/`<pop>` genre token that's currently reserved but unused.