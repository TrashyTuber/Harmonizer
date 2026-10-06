---
title: Harmonizer
emoji: 🎻
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
license: gpl-3.0
pinned: false
short_description: "Melody in, chords out: a BiLSTM trained on folk tunes"
---

# Harmonizer

Type or upload a melody and a bidirectional LSTM writes chords to accompany
it. The model was trained on the Nottingham folk dataset with every tune
transposed into all 12 keys, so melodies can be in any key.

Choose **Smoothed** decoding to have a Viterbi decoder with learned
chord-to-chord transitions clean up rapid chord changes, or **Raw** to hear
the model's best chord for each note on its own.

Training data: the [Nottingham Music Database](https://github.com/jukedeck/nottingham-dataset)
(Jukedeck's cleaned MIDI version), licensed under GPL-3.0.
