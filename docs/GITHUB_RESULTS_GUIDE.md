# GitHub Push & Result-Viewing Guide

This guide covers: (1) getting the final results onto GitHub so an examiner (or
you, on another machine) can **see the end result**, and (2) producing a
**readable speaker-attributed transcript** from the JSON output.

---

## Part A — What NOT to push (read this first)

Your `git status` currently shows untracked files. Before pushing, be careful:

**Do NOT commit real third-party face photos.** As of now these are untracked:
- `data/registry/Barrister_A_S_M_Shahriar_Kabir.jpg`
- `data/registry/Dr_Md_Tawohidul_Haque.jpg`
- `data/registry/Matiur_Rahman_Chowdhury.jpg`
- `data/registry/Zahed_Ur_Rahman.jpg`

These are real, identifiable public figures. The repo `.gitignore` currently
*un-ignores* registry images (`!data/registry/*.jpg`), so if you `git add .`
they will be committed. **Do not `git add` them.** Keep them on disk for your
local runs only, and document provenance instead (a `data/registry/README.md`).

**Do NOT commit the video or large intermediates** — `data/input/` and
`data/output/` are already gitignored. Your final results should live in a
**non-ignored** directory like `results/` so they CAN be pushed.

---

## Part B — Commit the code + tooling you want on GitHub

From the repo root:

```bash
cd ~/Developer/multimodal-speaker-indexing

# 1. Only add the code/docs/scripts, NOT the registry .jpg files.
git add docs/THESIS_RUN_BOOK.md docs/KAGGLE_RUN_BOOK.md
git add scripts/run_episode.py scripts/export_transcript.py
git add .gitignore
git status          # confirm the .jpg files are NOT staged

# 2. Commit (message in your own words is fine).
git commit -m "Add Kaggle/thesis run books and episode + transcript runners"

# 3. Push.
git push origin main
```

If you later want the (private) registry available for reproduction, put it in
a **private** repo or Kaggle Dataset and link it from the README — do not push
the photos to the public repo.

---

## Part C — Get your final results into the repo (so they're visible)

After a successful Kaggle run, download these from the notebook's **Output**
tab:
- `result.json` — the raw FinalSegment list
- `subtitles.srt` — speaker-labeled subtitles
- `health.json` — the fusion-health checks
- `run_<id>.json` — the reproducibility manifest

Put them in a **tracked** results folder (not `data/output`, which is ignored):

```bash
mkdir -p results/rtv_goll_table_ep
# copy the downloaded files there
cp .../result.json results/rtv_goll_table_ep/
cp .../subtitles.srt results/rtv_goll_table_ep/
cp .../health.json  results/rtv_goll_table_ep/
cp .../run_*.json   results/rtv_goll_table_ep/
```

Then generate the readable transcript (see Part D) and commit the folder:

```bash
git add results/
git commit -m "Add RTV Goll Table episode results + transcript"
git push origin main
```

---

## Part D — Produce a readable transcript (text + HTML)

The repo now has `scripts/export_transcript.py` which turns any `result.json`
into a clean, speaker-attributed transcript in **two formats**:

```bash
# Basic: writes <name>.txt and <name>.html next to the JSON
python scripts/export_transcript.py results/rtv_goll_table_ep/result.json

# Or choose a location/title
python scripts/export_transcript.py results/rtv_goll_table_ep/result.json \
    --out results/rtv_goll_table_ep/transcript --title "RTV Goll Table — transcript"
```

**Example output (text):**
```
[00:08.13 - 00:30.90] Abu Hena Razzaki  [conf=0.87]
    তোষামোধ নয় ক্ষমতাকে প্রশ্ন করাই গণমাধ্যমের কাজ আমরা প্রশ্ন করতে চাই ...

[00:32.16 - 00:38.59] Abu Hena Razzaki  [conf=0.87]
    আজকে আমরা যে বিষয়গুলো বলছি ...
```

The `.html` version is styled and is the best thing to **open in a browser** or
**share with your supervisor/examiner** — it shows the speaker name, timestamp,
and transcript text per segment.

---

## Part E — "See the end result" on GitHub (three ways)

1. **On github.com** — navigate to `results/rtv_goll_table_ep/` and click the
   `.html` transcript file: GitHub renders it as a readable page. The `.txt`
   also renders inline.
2. **Local HTML** — after `git clone`/`git pull`, open the `.html` file in any
   browser for the styled view.
3. **Raw SRT** — `subtitles.srt` is the speaker-labeled subtitle file you can
   open in a video player (rename to `.srt` next to the video) to see
   speaker-attributed captions over the actual footage.

---

## Part F — Full workflow summary (Kaggle → GitHub → readable)

```bash
# On Kaggle (see docs/KAGGLE_RUN_BOOK.md):
#   run the cells; download result.json, subtitles.srt, health.json, run_*.json
#   from the notebook Output tab.

# On your Mac:
mkdir -p results/rtv_goll_table_ep
# copy the downloaded files into results/rtv_goll_table_ep/

# Generate the transcript
python scripts/export_transcript.py results/rtv_goll_table_ep/result.json \
    --out results/rtv_goll_table_ep/transcript --title "RTV Goll Table transcript"

# Commit + push (code, docs, results — but NOT registry photos)
git add docs/ scripts/ results/ .gitignore
git status                 # verify no .jpg is staged
git commit -m "Add RTV Goll Table end-to-end results + transcript"
git push origin main

# View the result:
#   - open results/rtv_goll_table_ep/transcript.html in a browser, or
#   - open it on github.com after the push
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `git push` asks for credentials | Use a PAT or `gh auth login` (GitHub CLI), or push over SSH. |
| `result.json` not showing on GitHub | You put it in `data/output/` (gitignored). Move it to `results/`. |
| Registry `.jpg` staged by accident | `git reset` them, then re-add only the files you want. |
| `python scripts/export_transcript.py` not found | You're not in the repo root, or the file wasn't committed. |
| Want the SRT to show in a player | Name it exactly `<video>.srt` and place next to the video file. |
