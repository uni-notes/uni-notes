import shutil
from pathlib import Path
import glob


cwd = Path.cwd()
# attachment_folder = cwd / './docs'
# note_folder = cwd / './docs'
trash_folder = Path('./Trash')

from pathlib import Path

all_note_paths = (
    p.resolve() for p in Path("./").glob("**/*") if p.suffix in [
        ".md", ".css", ".js", ".html"
    ]
)

all_attachment_paths = (
    p.resolve() for p in Path("./").glob("**/*") if p.suffix in [
        ".jpg", ".jpeg", ".png", ".svg"
    ]
)

all_texts = ""

for note_path in all_note_paths:
    # pass
    with open(note_path, 'r') as f:
        all_texts +=  f.read()
# with open("test.txt", 'w') as f:
#         f.write(all_texts)
import os
for attachment_path in all_attachment_paths:
    name = os.path.basename(attachment_path)
    # print(attachment_path)
    if name not in all_texts:
        print(f'{name} moved to Trash')
        shutil.move(attachment_path, trash_folder/f'{name}')