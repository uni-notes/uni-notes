# CML

Continuous ML

## ML Model Report

```yaml
name: model-wine-quality
on: [push]
jobs:
  run:
    runs-on: [ubuntu-latest]
    container: docker://dvcorg/cml-py3:latest
    steps:
      - uses: actions/checkout@v2
      - name: cml_run
        env:
          repo_token: ${{ secrets.GITHUB_TOKEN }}
        run: |

          # Your ML workflow goes here
          pip install -r requirements.txt
          python train.py
          
          echo "## Model metrics" > report.md
          cat metrics.txt >> report.md
          
          echo "## Data viz" >> report.md
          cml-publish feature_importance.png --md >> report.md
          cml-publish residuals.png --md >> report.md
          
          cml-send-comment report.md
```

## DVC Pipeline for ML Model Report Compared to `main` branch

### `dvc.yaml`

```python
stages:
  get_data:
    cmd: python get_data.py
    deps:
    - get_data.py
    outs:
    - data_raw.csv  
  process:
    cmd: python process_data.py
    deps:
    - process_data.py
    - data_raw.csv
    outs:
    - data_processed.csv
  train:
    cmd: python train.py
    deps:
    - train.py
    - data_processed.csv
    outs:
    - by_region.png
    metrics:
    - metrics.json:
        cache: false
```

### `ci.yml`

```yaml
name: farmers
on: [push]
jobs:
  run:
    runs-on: [ubuntu-latest]
    container: docker://dvcorg/cml-py3:latest
    steps:
      - uses: actions/checkout@v2
      - name: cml_run
        env:
          repo_token: ${{ secrets.GITHUB_TOKEN }}
        run: |
          pip install -r requirements.txt
          dvc repro 

          git fetch --prune
          dvc metrics diff --show-md master > report.md

          # Add figure to the report
          echo "## Validating results by region"
          cml-publish by_region.png --md >> report.md
          cml-send-comment report.md
```