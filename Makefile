init:
	pip install -r requirements.txt

install:
	pip install -e .

lint:
	flake8 --extend-ignore=W605 .

test:
	python setup.py test