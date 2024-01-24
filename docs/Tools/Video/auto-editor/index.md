Always use `--keep-tracks-separate` to avoid weird syncing issues

## My usual settings

```bash
auto-editor input.mp4 --sounded-speed 1 --silent-speed 0 --edit "audio:stream=1,threshold=-30dB,mincut=1s" --my-ffmpeg --extras "-c:v hevc_nvenc -cq 30 -c:a copy -preset ultrafast" --keep-tracks-separate
```

## Check File Properties

```bash
auto-editor info test.mkv
```

## Default Settings

```bash
## all default settings
auto-editor test.mkv --show-ffmpeg-debug

auto-editor --scale --help
auto-editor --margin --help
```

## Output Stats Preview

```bash
auto-editor input.mp4 --stats
```

## Custom Output Name

```bash
auto-editor input.mp4 --output-file output.mp4
```

## Analyze different audio tracks

```bash
auto-editor multi-track.mov --edit "audio:stream=1 audio:threshold=10%" --keep-tracks-separate
auto-editor multi-track.mov --edit "(or audio:stream=0 audio:threshold=10%,stream=1)" --keep-tracks-separate
```

## GPU + CQP

```bash
#CQP
auto-editor video.mp4 --video-codec hevc_nvenc --my-ffmpeg --extras "-rc vbr_hq -qmin 0 -cq 30" --keep-tracks-separate

auto-editor test.mkv --video-codec hevc_nvenc --my-ffmpeg --keep-tracks-separate
auto-editor test.mkv --video-codec h264_nvenc --my-ffmpeg --keep-tracks-separate
```

## Premiere

```bash
auto-editor example.mp4 --export premiere --keep-tracks-separate
```

## Speed of quiet and loud parts

```bash
auto-editor video.mp4 --silent-speed 10.0 --sounded-speed 1.5 --keep-tracks-separate
```

## Analyze only a particular track

```bash
auto-editor video.mp4 --edit "audio:stream=2 audio:threshold=4%" --keep-tracks-separate

auto-editor video.mp4 --edit "(or audio:stream=2 audio:threshold=4%,stream=0)" --keep-tracks-separate
```

## Running from Python Script

### Process Single Video

```python
  import subprocess

  video = "test.mp4"

  command = r'auto-editor "{}" --video-codec hevc_nvenc --my-ffmpeg --keep-tracks-separate'.format(video)
  subprocess.Popen(command)
```

### Process Multiple Videos

```python
  import subprocess
  import os

  def find_videos(directory):
    videos = []
    for filename in os.listdir(directory):
      extension = filename.split(".")[-1]
      if(extension in ["mp4", "mkv"]):
        video = os.path.join(directory, filename)
        videos.append(video)
    return videos

  def process_video(video):
    command = r'auto-editor "{}" --video-codec hevc_nvenc --my-ffmpeg --keep-tracks-separate'.format(video)
    subprocess.Popen(command)

  if __name__ == "__main__":
    directory = os.getcwd()
    videos = find_videos(directory)

    for video in videos:
      process_video(video)
```

## Lossless

```python
import numpy as np
import time
import re
import sys
import json
import os
import concurrent.futures as multi

def cleanup():
  files_to_remove = []
  for file in os.listdir():
    for unwanted in ["_segment"]:
      if unwanted in file.lower():
        files_to_remove.append(file)
  with multi.ThreadPoolExecutor() as executor:
    executor.map(os.remove, files_to_remove)

def get_fps(input_file):
  info = os.popen(f'ffprobe -i "{input_file}" 2>&1').read()
  match = re.search(r'\s([\d\.]*)\sfps', info)
  if match:
    return float(match.group(1))
  return 0.0

def get_keyframe_interval(input_file):
  """
  Get average keyframe interval
  """

  start_time_to_read = 1
  max_seconds_to_read = 5
  info = os.popen(f"""
  ffprobe -read_intervals {start_time_to_read}%+{max_seconds_to_read} -select_streams v -show_entries frame=pts_time -of csv=p=0 -skip_frame nokey -v 0 -hide_banner -i {input_file}
  """).read()

  keyframe_time_points = np.array(info.split("\n"))
  keyframe_time_points = keyframe_time_points[
    (keyframe_time_points != "")
  ]
  keyframe_time_points = keyframe_time_points.astype(np.float32)
  keyframe_interval = np.round(np.mean(np.diff(keyframe_time_points))).astype(int)
  return keyframe_interval

def process_json(input_file, fps, json):
  extension = os.path.splitext(input_file)[1]
  cmd = []
  with open('_segments.txt', 'w') as f:
    sounded_chunks = json["v"][0]
    for segment_number, sounded_chunk in enumerate(sounded_chunks):
      segment_file_name = f"_segment{segment_number}{extension}"

      f.write(f'file {segment_file_name}\n')

      offset_time = sounded_chunk["offset"] / fps
      start_time = sounded_chunk["start"] / fps
      speed = sounded_chunk["speed"]
      duration = (sounded_chunk["dur"]/speed) / fps

      cmd.append(f"""
      ffmpeg -hide_banner -loglevel error \
      -ss {offset_time + start_time} \
      {(
        f"-itsscale {1/speed}"
        if speed==1
        else ""
      )} \
      -i "{input_file}" \
      -t {duration} \
      -avoid_negative_ts make_zero \
      {(
        "-c:a copy"
        if speed==1
        else f"-af volume=0 -af atempo={speed}"
      )} \
      -c:v copy \
      -map_metadata 0 -movflags use_metadata_tags -movflags '+faststart' -default_mode infer_no_subs -ignore_unknown -y \
      {segment_file_name}
      """)

    with multi.ProcessPoolExecutor() as executor:
      executor.map(os.system, cmd)

def combine_segments(input_file):
  os.system(f"""
  ffmpeg  -hide_banner -loglevel error \
  -f concat -safe 0 -protocol_whitelist 'file,pipe,fd' \
  -i _segments.txt \
  -c copy \
  '-disposition' default -movflags use_metadata_tags -movflags '+faststart' -default_mode infer_no_subs -ignore_unknown -y \
  "{os.path.splitext(input_file)[0]}_STRIPPED{os.path.splitext(input_file)[1]}"
  """)

def process_file(input_file):
  fps_val = get_fps(input_file)
  keyframe_interval = get_keyframe_interval(input_file)

  if fps_val <= 0.0:
    print(f'Unable to determine FPS of {input_file}')
    return

  os.system(f'auto-editor "{input_file}" --edit "audio:threshold=-40dB,mincut={max([1, 1 + keyframe_interval])}s" --export json')
  json_file = os.path.splitext(input_file)[0] + '_ALTERED.json'
  with open(json_file) as f:
    json_data = json.load(f)
  process_json(input_file, fps_val, json_data)
  os.remove(json_file)
  combine_segments(input_file)

def iterate_files():
  for input_file in sys.argv[1:]:
    process_file(input_file)

def main():
  cleanup()

  start_time = time.time()
  iterate_files()
  print("--- %s seconds ---" % (time.time() - start_time))

  cleanup()

if __name__ == "__main__":
  main()
```

## Editing logic

```bash
default: audio (only)

Editing Methods:
 - audio: General audio detection
 - motion: Motion detection specialized for real life noisy video
 - pixeldiff: Detect when a certain amount of pixels have changed between frames
 - random: Set silent/loud randomly based on a random or preset seed
 - none: Do not modify the media in anyway (Mark all sections as "loud")
 - all: Cut out everything out (Mark all sections as "silent")

Attribute Defaults:
 - audio
    - threshold: 4% (number)
    - stream: 0 (natural | "all")
 - motion
    - threshold: 2% (number)
    - stream: 0 (natural | "all")
    - blur: 9 (natural)
    - width: 400 (natural)
 - pixeldiff
    - threshold: 1 (natural)
    - stream: 0 (natural | "all")
 - random
    - threshold: 0.5 (number)
    - seed: RANDOMLY-GENERATED (int)

Operators:
 - and
   - usage: $METHOD and $METHOD
 - or
   - usage: $METHOD or $METHOD
 - xor
   - usage: $METHOD xor $METHOD
 - not
   - usage: not $METHOD
Examples:
  --edit audio
  --edit audio:stream=1
  --edit audio:threshold=4%
  --edit audio:threshold=0.03
  --edit motion
  --edit motion:threshold=2%,blur=3
  --edit audio:threshold=4% or motion:threshold=2%,blur=3
  --edit none
  --edit all
```
