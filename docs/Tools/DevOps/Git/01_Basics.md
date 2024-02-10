# Basics

## Cloning

### Sparse Checkout

```bash
git clone --filter=blob:none --sparse --no-checkout https://github.com/user_or_org/repo_name

# move to git folder
cd repo_name

# set sparse
# git sparse-checkout set --cone

# select branch
git checkout main

# clone specific folder
git sparse-checkout add ./folder

# clone specific file
git sparse-checkout add ./folder/file
```

### Shallow

```bash
git clone --single-branch --depth=1 --branch main https://github.com/user_or_org/repo_name
```
## Refresh Repo
Go to repo local folder
Right-click > `Git Bash here`
```bash
git checkout main
git checkout --orphan last
git add -A
git commit -am "Repo Refresh"
git branch -D main
git branch -m main
git gc --aggressive --prune=all
git push -f origin main

git checkout gh-pages
git checkout --orphan last
git add -A
git commit -am "Repo Refresh"
git branch -D gh-pages
git branch -m gh-pages
git gc --aggressive --prune=all
git push -f origin gh-pages
```
## Rebase
```bash
git rebase -i HEAD~10
```

## Rename/Moving File

```bash
git mv folder_old/filename_old folder_new/filename_new
```

This avoids Git thinking that the files were deleted & added, thus reducing repo size

