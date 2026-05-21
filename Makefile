.PHONY: help setup-f1 lint clean-nb

help:
	@echo "Available targets:"
	@echo "  setup-f1   install dependencies for the F1 pit stops experiment"
	@echo "  lint       run ruff over Python sources"
	@echo "  clean-nb   strip outputs from all notebooks"

setup-f1:
	pip install --system -r experiments/kaggle-f1-pit-stops/requirements.txt

lint:
	ruff check .

clean-nb:
	find . -name "*.ipynb" -not -path "./.git/*" -exec jupyter nbconvert --clear-output --inplace {} +
