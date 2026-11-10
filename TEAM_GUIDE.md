# Team guide: where everything goes

Hi team! It's Sheja. The data part is done and the repo is set up at https://github.com/Queenlinda12/formative2-multimodal-auth (accept the invite in your email). Here's what each of us needs to do next, so we all use the same names and folders.

## Everyone (please do this first)

Send your files to B and C, or push them yourself, using **exactly** these names (all lowercase, your first name):

| What | Folder | File name |
| --- | --- | --- |
| Face, neutral | `images/raw/` | `yourname_neutral.jpg` |
| Face, smiling | `images/raw/` | `yourname_smiling.jpg` |
| Face, surprised | `images/raw/` | `yourname_surprised.jpg` |
| Voice, "Yes, approve" | `audio/raw/` | `yourname_yesapprove.wav` |
| Voice, "Confirm transaction" | `audio/raw/` | `yourname_confirmtransaction.wav` |

Tips: take the photos in good light, face the camera, no sunglasses. Record the audio somewhere quiet, about 2 to 3 seconds per clip. If your phone saves `.m4a`, that's fine, but B or C will convert it to `.wav`.

## Member B (Image lead)

- Raw photos go in `images/raw/` and augmented ones in `images/augmented/`
- Save features to `features/image_features.csv` (with columns `member`, `expression`, `augmentation`)
- Save the model in `models/` and write `predict_face(image_path)` in `scripts/`. It should return the member's name or `"unknown"`
- Add one photo of someone outside the team as `images/raw/unauthorized_face.jpg`

## Member C (Audio lead)

- Raw clips go in `audio/raw/` and augmented ones in `audio/augmented/`
- Save features to `features/audio_features.csv` (with columns `member`, `phrase`, `augmentation`)
- Save the model in `models/` and write `verify_voice(audio_path)` in `scripts/`. It should return the member's name or `"unauthorized"`
- Add one clip of a voice outside the team as `audio/raw/unauthorized_voice.wav`
- Finish `README.md` (I started a draft)

## Member D (Product model + app)

- Train on `data/processed/merged_dataset_model_ready.csv`
  - Inputs: every column except `transaction_id`, `customer_id`, `product_category` and `target_code`
  - Label: `target_code` (0 Books, 1 Clothing, 2 Electronics, 3 Groceries, 4 Sports)
- Use `GroupKFold` grouped by `customer_id`, because the same customer appears in several rows
- Heads up: my quick Random Forest test got about 37% accuracy (random guessing gets 24%). The data just has a weak pattern, so low scores are expected, not a bug
- Write `predict_product(...)` and build `scripts/main.py`: face → voice (must be the **same person**) → show the product

## Report and video

- Write your own report section (about 1 page) with what you built and your results, and send it to me (Sheja). I'll put the final PDF together
- Send me your row for the contribution table: your name, role, what you built
- Record your 1 to 2 minute screen recording (mp4) and send it to D

Commit your own work with clear messages, so everyone's contribution shows in the repo.

Thanks everyone!
Sheja
