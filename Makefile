#
# Simple orchestration for major tests
#

.PHONY: bert_tests

bert_tests:
	python src/text_models/bert.py \
		-d fashion_mnist \
		-n 100 \
		-t 30 \
		-o calc/distilbert_fashionmnist_n100_t30.csv
	python src/text_models/bert.py \
		-d fashion_mnist \
		-n 1000 \
		-t 30 \
		-o calc/distilbert_fashionmnist_n1000_t30.csv
	python src/text_models/bert.py \
		-d fashion_mnist \
		-n 10000 \
		-t 30 \
		-o calc/distilbert_fashionmnist_n10000_t30.csv
