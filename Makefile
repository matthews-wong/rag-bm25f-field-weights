.PHONY: test eval validate

test:
	python3 -m unittest

eval:
	python3 -m bm25f.evaluate

validate: test eval
