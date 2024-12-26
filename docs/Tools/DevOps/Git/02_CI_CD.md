# CI/CD

## Actions

Create `~/.github/workflows/ci.yml`

```yaml
name: Deploy uni-notes using Mkdocs
on:
  push:
    branches:
      - main
jobs:
  deploy:
    runs-on: ubuntu-latest
    env:
      MKDOCS_GIT_COMMITTERS_APIKEY: ${{ secrets.GITHUB_TOKEN }}
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: '0'
      - uses: actions/setup-python@v3
        with:
          python-version: 3.x
          cache: pip
      - run: pip install -r requirements.txt
      - run: mkdocs gh-deploy --force --no-history
```

```yaml
name: Scrape otexts.com/fpp3
on:
  schedule:
    - cron: '45 9 * * 3'  # Runs every Wednesday at 09:45 UTC
  workflow_dispatch:
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: '0'
      - name: Check for 'docs' branch and create if it doesn't exist
        run: |
          git fetch origin
          if git show-ref --verify --quiet refs/heads/docs; then
            echo "Branch 'docs' already exists."
          else
            echo "Creating 'docs' branch."
            git checkout -b docs
            git push origin docs
          fi
      # - name: Create output directory
      #   run: |
      #     mkdir -p ./docs # This won't error if the directory already exists
      #     echo "Directory created: $(ls -d .)"  # Confirm directory creation
      - name: Install wget
        run: sudo apt-get install -y wget
      - name: Perform Scraping
        run: wget --no-parent -r -l 2 -P . "https://otexts.com/fpp3"
        continue-on-error: true
      - name: Create output directory
        run: |
          cd otexts.com
          echo "Moving to otexts.com: $(ls -d .)" # Confirm cd
      - name: Configure Git
        run: |
          git config --local user.name "AhmedThahir"
          git config --local user.email "ahmedthahir2002@gmail.com"
      - name: Commit changes
        run: |
          git checkout docs
          git add .
          git commit -m "Add current directory" || echo "No changes to commit"
      - name: Push changes
        run: |
          git push origin docs
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## DVC

Data Version Control

You can use an external storage for non-code files.

Especially useful for large files

