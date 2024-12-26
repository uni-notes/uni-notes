## Run program without admin privileges

Be careful; this does not eliminate the need for authorization. Never run any programs without authorization.

Useful to run run program with low privileges

### `run_{program_name}.bat`

```bash
Set __COMPAT_LAYER=RunAsInvoker
Start program_name.exe
```
