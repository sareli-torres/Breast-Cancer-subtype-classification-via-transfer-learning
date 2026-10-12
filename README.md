# Breast Cancer Subtype Classification via Transfer Learning

DATA7703 Machine Learning for Data Scientists — Group 20

We train transfer learning models (ResNet50, EfficientNet-B0, VGG16) and a baseline CNN on the **BreaKHis** dataset for two tasks:

- **Binary:** benign vs. malignant
- **8-class:** adenosis, fibroadenoma, phyllodes tumor, tubular adenoma, ductal carcinoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma

Research questions:

1. Do models trained on one magnification (40X, 100X, 200X, 400X) generalise to the others?
2. When accuracy drops under magnification shift, does the model's reasoning (Grad-CAM) break down too?

---

## Who does what

| Role | Owner(s) | Files you edit | Report section |
|---|---|---|---|
| 1. Data | Sareli, Mandara | `src/make_metadata.py`, `src/data.py` | Data and preprocessing |
| 2. Training pipeline | Mandara, Sareli | `src/models.py`, `src/train.py` | Method and training setup |
| 3. ResNet50 experiments | Dhruv | `notebooks/` (your own notebook) | Results |
| 4. EfficientNet-B0 and VGG16 experiments | Peehoo | `notebooks/` (your own notebook) | Introduction and related work |
| 5. Grad-CAM and analysis | Sonal | `src/gradcam.py` | Explainability, discussion |

**Golden rule: only edit your own files.** If you need a change in someone else's file, ask them. This is how we avoid overwriting each other's work.

---

## Where things live

| Place | What it stores |
|---|---|
| **GitHub** (this repo) | Code only |
| **Google Drive** (`DATA7703_G20` shared folder) | Dataset, checkpoints, results, figures |
| **Google Colab** | Nothing — it only runs the code |

Never commit data, checkpoints or images to GitHub. The `.gitignore` already blocks `data/`, `checkpoints/`, `*.pt` and `*.tar.gz`.

The dataset (`BreaKHis_v1.tar.gz`, 4 GB) is **already in the shared Drive folder**. Do not download it again.

---

## One-time setup (everyone, ~15 min)

### 1. Accept the GitHub invitation

Check your email or https://github.com/notifications and accept the invitation to this repo.

### 2. Add the shared Drive folder to your Drive

1. In Google Drive, open **Shared with me**.
2. Right-click `DATA7703_G20` → **Organise → Add shortcut**.
3. Choose **My Drive** (the top level) and confirm.

This step is required: it makes the folder path the same for everyone in Colab:
`/content/drive/MyDrive/DATA7703_G20`

### 3. Create a GitHub token

The repo is private, so Colab needs a token to download the code.

1. On GitHub, click your profile photo → **Settings**.
2. At the bottom of the left menu, click **Developer settings**.
3. **Personal access tokens → Tokens (classic) → Generate new token (classic)**.
4. Note: `colab` · Expiration: 30 days · Scopes: tick **repo** only.
5. Click **Generate token** and **copy it** (starts with `ghp_`). GitHub only shows it once.

Never paste the token into a notebook cell, the group chat, or a commit.

### 4. Save the token in Colab

1. Open Colab and create a notebook.
2. Click the **key icon** 🔑 in the left panel (Secrets).
3. **+ Add new secret** → Name: `GITHUB_TOKEN` (exactly) → Value: paste your token.
4. Turn on **Notebook access**.

The secret is saved to your Colab account. You only need to turn on **Notebook access** again in each new notebook.

### 5. Test it

In Colab, select **Runtime → Change runtime type → T4 GPU**, then run these cells:

```python
# Cell 1 — check the GPU (should show "Tesla T4")
!nvidia-smi
```

```python
# Cell 2 — connect Google Drive
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = "/content/drive/MyDrive/DATA7703_G20"
!ls "{DRIVE_ROOT}"
```

```python
# Cell 3 — download the code from GitHub
from google.colab import userdata
token = userdata.get('GITHUB_TOKEN')
!git clone https://{token}@github.com/sareli-torres/Breast-Cancer-subtype-classification-via-transfer-learning.git repo
%cd repo
!ls
```

```python
# Cell 4 — copy the dataset to Colab's local disk and extract it (a few minutes)
!cp "{DRIVE_ROOT}/data/BreaKHis_v1.tar.gz" /content/
!tar -xzf /content/BreaKHis_v1.tar.gz -C /content/
!ls /content/BreaKHis_v1
```

If all four cells work, you're set up. Post a ✅ in the group chat.

---

## Every time you work

Each Colab session starts empty, so run cells 1–4 again at the start of every session. Your results are safe because they are saved to Drive.

If the repo is already cloned in the current session, update it instead of cloning again:

```python
%cd /content/repo
!git pull
```

We copy the data to `/content` instead of reading it from Drive because reading thousands of small images from Drive is very slow.

---

## How to change code

**Option A — on your computer (recommended):**

```bash
git clone https://github.com/sareli-torres/Breast-Cancer-subtype-classification-via-transfer-learning.git
git pull                          # every time, before you start
# ...edit your files...
git add .
git commit -m "short description of the change"
git push
```

**Option B — on the GitHub website:** open the file, click the pencil icon, edit, then **Commit changes**.

Then run `!git pull` in Colab to get the latest code.

---

## Naming conventions

Use exactly these names so everyone can find each other's outputs:

| Output | Drive folder | Name pattern | Example |
|---|---|---|---|
| Checkpoint | `checkpoints/` | `{model}_{task}_{mag}.pt` | `resnet50_binary_40x.pt` |
| Results | `results/` | `{model}_{task}_{mag}.json` | `resnet50_binary_40x.json` |
| Figures | `figures/` | descriptive name | `gradcam_resnet50_40x_on_400x.png` |

- `model`: `resnet50`, `efficientnet_b0`, `vgg16`, `baseline_cnn`
- `task`: `binary`, `8class`
- `mag`: `40x`, `100x`, `200x`, `400x`, `all`

One results file per run (never a shared file everyone appends to — Drive will overwrite changes).

---

## Repo structure

```
src/
  make_metadata.py   builds metadata.csv (one row per image) and the patient-level split
  data.py            dataset class, transforms, dataloaders
  models.py          ResNet50, EfficientNet-B0, VGG16 and baseline CNN
  train.py           trains one model and evaluates it on all 4 magnifications
  gradcam.py         Grad-CAM heatmaps
notebooks/
  run_experiment.ipynb   Colab template
```

Key experiment decisions (fixed for every run):

- **Patient-level split**, identical across magnifications, with at least one patient of every subtype in validation and test. The split is saved in `metadata.csv` and is **never regenerated** once frozen.
- 224×224 input, ImageNet normalisation, same augmentations, frozen backbone with a new classification head, class-weighted cross-entropy, early stopping on validation macro-F1, fixed seed.
- Validation uses the **same magnification as training**. Test is run on **all four** magnifications.

---

## Timeline

| Date | Milestone |
|---|---|
| Mon 12 Oct | Setup done for everyone; `metadata.csv` and split frozen |
| Tue 13 Oct | First full run (ResNet50, binary, all magnifications); run time measured |
| Wed 14 – Thu 15 Oct | All experiment runs; Grad-CAM on ResNet50 checkpoints |
| Fri 16 Oct | **Experiment freeze** — final figures, everyone drafts their section |
| Sat 17 Oct | Full first draft assembled |
| Sun 18 Oct | Cross-review: check every number against the results files |
| Mon 19 Oct | Final read and submission |

If we fall behind, we cut in this order (bottom first): baseline 8-class → VGG16 → EfficientNet-B0 → baseline binary. ResNet50 (both tasks) plus Grad-CAM is the minimum that answers both research questions.

---

## Troubleshooting

| Error | Likely cause | Fix |
|---|---|---|
| `SecretNotFoundError` | Secret name misspelled or **Notebook access** is off | Check the key icon 🔑: name must be `GITHUB_TOKEN`, toggle on |
| `Repository not found` | Invitation not accepted, or token missing the **repo** scope | Accept the invite; regenerate the token with **repo** ticked |
| `already exists and is not an empty directory` | Repo already cloned in this session | Run `%cd /content/repo` then `!git pull` |
| `No such file or directory` with `DATA7703_G20` | Drive shortcut not added to **My Drive** | Redo setup step 2 |
| Session disconnected mid-run | Colab timeout | Rerun cells 1–4; finished runs are already on Drive |

---

## Dataset citation

Spanhol, F., Oliveira, L. S., Petitjean, C., Heutte, L. (2016). A Dataset for Breast Cancer Histopathological Image Classification. *IEEE Transactions on Biomedical Engineering*, 63(7), 1455–1462.
