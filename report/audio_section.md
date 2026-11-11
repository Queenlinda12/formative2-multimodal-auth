# Audio Data and Voiceprint Verification Model

**Agnes Mbabazi (Audio lead)**
GitHub @agmbabazi

I handled the voice part of the system: collecting and checking the recordings, augmenting them, extracting features and training the voiceprint model.

## Collection and augmentation

Each member recorded "Yes, approve" and "Confirm transaction" on their phone, saved as `firstname_phrase.m4a`, with extra takes as `firstname_phrase_2.m4a`. My script converts every recording to the same format (16 kHz mono WAV), trims the silence at the start and end, and scales every clip to the same volume, so the model hears the voice rather than the pause length or how close the phone was. It also checks that every member has both phrases, that names follow the format, that no recording is distorted, and that no file is an exact copy of another. That last check matters: when copied files were renamed as new takes, accuracy jumped to a fake 100%, because the same recording ended up in both training and testing. For the demo stranger, someone outside the team recorded both phrases.

I applied five augmentations to every recording: pitch up and down by one semitone, 10% faster, 10% slower, and light background noise. I measured how far each one moves a voice and kept them smaller than the gap between two members. Stronger settings (four semitones, louder noise) turned one member's voice into "someone else".

## Features

For every half-second of each clip I extracted 13 MFCCs (the shape of the voice), spectral roll-off and centroid (how bright it sounds), zero-crossing rate and RMS energy, as averages and spreads: 34 numbers per row. I used short windows because one average per clip mostly described which words were said, and scored no better than guessing. All 688 rows (16 recordings, each with 5 augmentations) are saved in `audio_features.csv`, with columns for member, phrase, augmentation and source recording.

## Model and results

I trained two Logistic Regressions: one for who is speaking, and one for which phrase. To test fairly, I always held out a whole recording together with its augmented copies.

On held-out recordings the speaker model scored 68% accuracy, an F1 score of 0.59 and a log loss of 1.30 (guessing among four people gives 25%). The biggest factor was data: members with three takes per phrase were recognised far more often (Naillah 92%) than members with one (Aime 0%). More real recordings helped more than any change of features or model I tried. Aime's and Sheja's extra takes are still coming, and I'll re-check the results once they're in.

A model like this always names a member, even for a stranger. So a voice is only accepted if the model is confident enough. The script picks that threshold automatically, where strangers let in (FAR) and members turned away (FRR) are equal: currently 0.60, with both about 0.56. The real stranger was rejected in all 12 versions of their clips.

## Use in the app

The app calls `verify_voice(audio_path)`, which returns the member's name or "unauthorized". It's the second gate, and it must match the face.

## My contribution

| Member | Role | What I built |
|---|---|---|
| Agnes Mbabazi | Audio lead | Audio collection checks (including copy detection), augmentations, `audio_features.csv`, the voiceprint model and `verify_voice`, the audio notebook, and the README |
