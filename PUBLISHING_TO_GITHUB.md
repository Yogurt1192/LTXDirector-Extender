# Publishing To GitHub

This repo is meant to be published as a fork-derived project that clearly credits the original upstream author.

## Recommended Flow

1. Open the upstream repo: <https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI>
2. Click **Fork** and choose the `Yogurt1192` account.
3. If GitHub does not let you use `LTXDirector-Extender` during the fork flow, create the fork first and then rename it in the repo settings.
4. After the GitHub repo exists, connect this local clone to your fork.

## Local Commands

Run these from this folder:

```bash
cd /media/jjbai1315/locker/Code/LTXDirector-Extender

git remote rename origin upstream
git remote add origin git@github.com:Yogurt1192/LTXDirector-Extender.git
git remote -v

git add README.md PUBLISHING_TO_GITHUB.md pyproject.toml ltx_director_guide.py prompt_relay.py
git commit -m "Publish LTX Director extension fixes"
git push -u origin main
```

## What `upstream` And `origin` Mean

- `upstream` = the original WhatDreamsCost repo
- `origin` = your GitHub repo under `Yogurt1192`

That layout makes it easy to keep credit clear and pull future upstream changes when needed.

## Pulling Future Upstream Changes

```bash
cd /media/jjbai1315/locker/Code/LTXDirector-Extender
git fetch upstream
git merge upstream/main
```

Resolve conflicts carefully in the LTX Director files if upstream changes the same extension logic.

## Important Note About PromptRelay Temporal LoRA

This fork now includes a standalone `prompt_relay_lora.py` node with Temporal LoRA controls.

It is still separate from the tested LTX Director extension path. Treat it as experimental until you validate it specifically with an LTX Director extension workflow.