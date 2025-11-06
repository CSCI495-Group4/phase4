###**** team4_models directory has both notebooks as we turned in for end of phase 3.


# Phase 4

This guide explains how to **access, download, edit, and upload** changes to this repository.  
It covers cloning the repo, setting up SSH keys, working with Jupyter Notebooks (`.ipynb`) and Python scripts (`.py`), and collaborating through Git branching and Pull Requests (PRs).

---

## 📌 Repo URL

**SSH Clone URL:**  
`git@github.com:CSCI495-Group4/phase3.git`

---

## 0) Prerequisites

- [Git](https://git-scm.com/) installed (`git --version`)
- SSH key set up with GitHub (instructions below)
- Editor of your choice:
  - **Jupyter Notebook / JupyterLab** or **VS Code with Jupyter extension** (for `.ipynb`)
  - Any Python IDE/editor (for `.py`)

---

## 1) Generate & Add an SSH Key (One-Time Setup)

Check if you already have a key 
```
ls -al ~/.ssh
```

If you see id_rsa.pub, id_ed25519.pub, or similar, you may already have a key.

Generate a new key (if needed)

```
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press Enter to save to default path (~/.ssh/id_ed25519)

Optionally set a passphrase (or leave blank)

Start SSH agent and add key

```
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

Add public key to GitHub
```
cat ~/.ssh/id_ed25519.pub
```

Copy the output.
On GitHub: Settings → SSH and GPG Keys → New SSH Key → Paste → Save

Test connection
```
ssh -T git@github.com
```

Should return:

Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.

## 2) Clone the Repository
# Navigate where you want to store the repo
```
cd ~/code
```
# Clone with SSH
git clone git@github.com:CSCI495-Group4/phase3.git
cd phase3

# Set your Git identity (once per machine)
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"

## 3) Branching Model

We use feature branches off main, then merge via Pull Requests.

Branch name conventions:

feat/<topic> → new functionality

fix/<topic> → bug fix

docs/<topic> → documentation only

Example:

# Create a new branch
```
git checkout -b feat/data-cleaning
```

Sync with latest main:

git fetch origin
git pull --rebase origin main

## 4) Working With Files
Option A: Jupyter Notebooks (.ipynb)

Open using JupyterLab, Jupyter Notebook, or VS Code with Jupyter extension

Save regularly

Consider using nbstripout to remove metadata (see Appendix A)

Option B: Convert Notebook to Python Script (.py)

Convert to plain .py for editing in any IDE:

```
jupyter nbconvert --to script notebook.ipynb
```

This creates notebook.py.

You can keep both .ipynb (for interactive runs) and .py (for cleaner diffs).

## 5) Add, Commit, and Push
# Check status
```
git status
```
# Stage changes
```
git add file.ipynb
git add file.py
```
# or: git add -A  (stage all)
# Commit with a clear message
```
git commit -m "feat: add baseline training notebook"
```
# Push branch to GitHub
```
git push -u origin feat/data-cleaning
```
## 6) Open a Pull Request (PR)

Go to the repo on GitHub: CSCI495-Group4/phase3

You’ll see a prompt to open a PR for your branch

Fill out:

Title (short summary)

Description (what/why/how to test)

Assign reviewers

After approval → Merge (prefer “Squash & merge”)

## 7) Sync Local After Merges

When someone merges to main:

# Update local main
```
git checkout main
git fetch origin
git pull origin main
```
# Update your feature branch
```
git checkout feat/data-cleaning
git rebase main
git push --force-with-lease
```
## 8) Common Fixes

Error: src refspec main does not match any
You don’t have a commit yet:
```
git add .
git commit -m "initial commit"
git branch -M main
git push -u origin main
```

Accidentally added big files
Add to .gitignore (see Appendix B) and remove from repo

Repo already has README or commits
```
git fetch origin
git pull --rebase origin main
git push -u origin <branch>
```
🔑 Quick Commands Reference
# New branch
```
git checkout -b feat/my-change
```
# Stage & commit
```
git add -A
git commit -m "feat: my change"
```
# Push to GitHub
```
git push -u origin feat/my-change
```
# Rebase with main
```
git fetch origin
git pull --rebase origin main
```
## Appendix A: Cleaner Notebook Diffs (Optional)

To strip output/metadata before commit:
```
pip install nbstripout
nbstripout --install
```
## Appendix B: Suggested .gitignore
# Byte-compiled / cache
__pycache__/
*.py[cod]
*.pyo

# Virtual environments
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/
*.nbconvert.ipynb

# OS / editor
.DS_Store
Thumbs.db
.vscode/
.idea/

# Data / outputs
data/
datasets/
outputs/
logs/
checkpoints/
*.pth
*.pt
*.ckpt

## Appendix C: Converting .ipynb ↔ .py

Notebook → Script

jupyter nbconvert --to script notebook.ipynb


Keep both if useful:

.ipynb for interactive runs

.py for reviews & clean version control

## Appendix D: Commit Message Tips

Format:

<type>: <short summary>


Types: feat, fix, docs, refactor, test, chore

Examples:

feat: add ResNet18 baseline

fix: resolve DataLoader shuffle bug

docs: update README with SSH key setup
