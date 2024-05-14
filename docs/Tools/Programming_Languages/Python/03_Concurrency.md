# Concurrency

## Async



## Multi-Threading



## Multi-Processing



## Subprocess

```python
for chunk in range(1, 10):
		command = f"command -- {chunk}"

		command_list = shlex.split(command)
		procs.append(
			subprocess.Popen(
        command_list,
        shell=True,
        text=True
      )
		)

		time.sleep(3)
    # some gap between starting up subprocesses

for p in procs:
  p.wait()