#
# Simple orchestration for major tests
#

.PHONY: bert_tests

bert_tests:
	python src/text_models/bert.py -d fashion_mnist -n 100 -o calc/training_history.csv
