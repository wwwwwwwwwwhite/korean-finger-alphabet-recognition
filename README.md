# Son — Korean Finger Alphabet Lab

A local-first web app for collecting Korean fingerspelling samples, training a
Random Forest classifier, and trying real-time recognition from a webcam. It
recognizes only the configured basic Korean consonant and vowel labels. It does
not translate sentences or use ASL/JSL label mappings.

## Important: gestures, data, and expected accuracy

The National Institute of Korean Language's [Korean Sign Language Dictionary
entry for 지문자](https://sldict.korean.go.kr/front/sign/signContentsView.do?origin_no=8087&top_category=CTE&category=&searchKeyword=%EC%A7%80%EB%AC%B8%EC%9E%90&searchCondition=&search_gubun=&museum_type=00&current_pos_index=0)
and [한글 지문자 entry](https://sldict.korean.go.kr/front/sign/signContentsView.do?origin_no=1383&top_category=CTE&category=&searchKeyword=%EC%A7%80%EB%AC%B8%EC%9E%90&searchCondition=&search_gubun=&museum_type=00&current_pos_index=1)
are linked in the Learn Letters page. They establish the Korean terminology,
but the verified entries do not provide a complete letter-by-letter pose chart
or establish which of the requested letters require movement. The app
intentionally does **not** make up pose instructions. Check an authoritative
Korean fingerspelling chart before collecting each label. If the standard
requires a moving gesture for a letter, this first version's static 21-point
frame classifier cannot represent that temporal gesture; collect only verified
static forms until sequence modeling is added.

No verified, clearly licensed Korean fingerspelling dataset or ready-to-use
model for this exact MediaPipe + scikit-learn pipeline was identified:

- The [pgu1458 Korean fingerspelling project](https://github.com/pgu1458/Korean-Sign-Language-Fingerspelling-Input-System)
  is a useful implementation reference, but its README claims are not backed by
  an included training dataset/model, the class listing has a duplicate vowel,
  and no explicit license was found. It is not bundled or treated as a
  pretrained model.
- The [Kaggle 14-consonant dataset](https://www.kaggle.com/datasets/ttr9904ewhaincom/korean-sign-language14-consonant-letters)
  is image-based and has an unknown license.
- Other reviewed resources either did not verify fingerspelling labels or use
  a different task/landmark representation.

Accordingly, this project starts with an **empty dataset and no trained
classifier**. Live predictions remain unavailable until you collect data and
train. No random or fabricated predictions are shown. Recognition quality
depends on the quality and diversity of the user's labeled captures.

## Features

- React + TypeScript + Vite frontend.
- MediaPipe Tasks Vision hand tracking in the browser; draws and sends the
  21 hand landmarks' `x`, `y`, and `z` values. Camera video is not uploaded.
- FastAPI backend with CSV-backed capture, category counts, sample deletion,
  and CSV export.
- Landmark centering/scaling normalization, same-class near-duplicate
  rejection, and a scikit-learn Random Forest classifier.
- Training metrics: accuracy, macro precision/recall/F1, per-letter
  precision/recall/F1, and a confusion matrix.
- The evaluation uses a stratified sample-level holdout. Since capture sessions
  are not held out as groups, correlations between distinct captures can still
  make metrics optimistic; the report displays this caveat. Use varied captures
  and do not treat a small test score as a real-world guarantee.
- Real-time prediction confidence threshold and recent-frame smoothing.
- Automatic or manual letter confirmation, undo, clear, and Unicode Hangul
  syllable composition (initial, medial, and final consonant), including
  sequential compound vowels such as `ㅗ+ㅏ=ㅘ`, `ㅜ+ㅓ=ㅝ`,
  `ㅜ+ㅔ=ㅞ`, and `ㅡ+ㅣ=ㅢ`.

### Recognized labels

Consonants: `ㄱ ㄴ ㄷ ㄹ ㅁ ㅂ ㅅ ㅇ ㅈ ㅊ ㅋ ㅌ ㅍ ㅎ`

Vowels: `ㅏ ㅑ ㅓ ㅕ ㅗ ㅛ ㅜ ㅠ ㅡ ㅣ ㅐ ㅔ`

Every class must have at least **3 distinct accepted samples** before training
(two training examples plus at least one held-out example per class). With all
26 labels this means at least 78 accepted examples. Near-identical normalized
landmarks for the same label are rejected; move or reorient the hand between
captures. This minimum is not a promise of good accuracy.

## Requirements

- macOS on Apple Silicon or Intel.
- Python 3.9 or later.
- Node.js 18 or later and npm.
- Internet for initial npm/pip dependency installation and MediaPipe's browser
  WASM/model asset download. The API and captured dataset run locally.
- A webcam and a browser camera permission for `localhost`.

## Quick start (macOS)

From the project directory:

```bash
chmod +x start.sh
./start.sh
```

The script creates `.venv`, installs Python and npm dependencies if needed,
starts FastAPI and Vite, and opens `http://localhost:5173`. Press `Ctrl+C` to
stop both servers.

### Manual setup

Terminal 1 — backend:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Terminal 2 — frontend:

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The Vite dev server proxies
`/api` calls to `http://127.0.0.1:8000`. FastAPI's interactive API docs are at
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Typical workflow

1. Open **Data Collection**, start the camera, choose a label, and begin
   capture. Hold the verified Korean pose and vary the hand's position or
   orientation; near-duplicate samples are skipped. Repeat for all 26 labels.
2. Use the per-letter counts to find missing or underrepresented classes.
   Review/delete bad samples and use **Export CSV** for a backup.
3. Open **Model Training** and train after every label has at least three
   accepted samples. Review the holdout scores, per-letter recall, and
   confusion matrix.
4. Open **Live Recognition**, allow camera access, and choose a confidence
   threshold. **Auto-add stable letters** is on by default: a prediction is
   appended once after it remains stable and above threshold for about 700 ms.
   The same held pose will not repeat; show another letter or briefly remove
   your hand to re-arm it for repeated letters. Turn auto-add off to use the
   manual **Confirm letter** button instead.

Adding or deleting samples invalidates the saved model, so retrain before
recognition. Captured landmarks are stored in `data/dataset.csv`; the classifier
is stored in `data/random_forest.joblib`. Both are local and ignored by Git.
Back up the CSV if you want to preserve your collection.

## API overview

- `GET /api/health`
- `GET /api/labels`
- `GET /api/samples` and `GET /api/samples/summary`
- `POST /api/samples` with a supported label, 21 `{x,y,z}` points, and optional
  `session_id`
- `DELETE /api/samples/{sample_id}`
- `GET /api/samples/export`
- `POST /api/train`
- `POST /api/predict` with 21 `{x,y,z}` points

`POST /api/train` returns `409` with per-label counts until every supported
label meets the minimum. `POST /api/predict` returns `404` until a model has
been trained. Dataset changes invalidate the saved model.

## Development checks

```bash
npm run build
source .venv/bin/activate
pytest
```
