export PYTHONPATH := 'scr'
export PIPENV_VERBOSITY := -1

setup:
	@sudo modprobe -r v4l2loopback
	@sudo modprobe v4l2loopback devices=1 exclusive_caps=1 video_nr=2 card_label='Virtual Webcam'

run:
	@pipenv run python src/apps.py

lint:
	@pipenv run black .
	@pipenv run isort .

qa:
	@pipenv run pytest

# ls -1 /sys/devices/virtual/video4linux
