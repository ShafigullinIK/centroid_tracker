env:
	conda create --name centroid_tracker --file requirements.txt

activate:
	conda activate centroid_tracker

demo:
	python3 app0.py --input input/test.mp4 --output output/my_result.mp4