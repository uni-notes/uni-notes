# Flask

```python
from flask import (
    Flask,
    request,
	send_file
)

app = Flask(__name__)

@app.route('/badge-going')
def return_badge_going():
	# badge_type = request.args.get('badge_type')
	file_name = "badge_going.jpg"
	try:
		return send_file(
		    f"./{file_name}",
		    # attachment_filename = file_name
		)
	except Exception as e:
		return str(e)
```

