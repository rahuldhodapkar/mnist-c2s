#
# Simple orchestration for major tests
#

.PHONY: all bert_tests resnet_tests cvt_tests

bert_tests:
	mkdir -p calc/image_classification
	python src/text_models/bert.py \
		-d fashion_mnist \
		-n 100 \
		-t 30 \
		-o calc/image_classification/distilbert_fashionmnist_n100_t30.csv
	python src/text_models/bert.py \
		-d fashion_mnist \
		-n 1000 \
		-t 30 \
		-o calc/image_classification/distilbert_fashionmnist_n1000_t30.csv
	python src/text_models/bert.py \
		-d fashion_mnist \
		-n 10000 \
		-t 30 \
		-o calc/image_classification/distilbert_fashionmnist_n10000_t30.csv
	python src/text_models/bert.py \
		-d mnist \
		-n 100 \
		-t 30 \
		-o calc/image_classification/distilbert_mnist_n100_t30.csv
	python src/text_models/bert.py \
		-d mnist \
		-n 1000 \
		-t 30 \
		-o calc/image_classification/distilbert_mnist_n1000_t30.csv
	python src/text_models/bert.py \
		-d mnist \
		-n 10000 \
		-t 30 \
		-o calc/image_classification/distilbert_mnist_n10000_t30.csv
	python src/text_models/bert.py \
		-d cifar10 \
		-n 100 \
		-t 30 \
		-o calc/image_classification/distilbert_cifar10_n100_t30.csv
	python src/text_models/bert.py \
		-d cifar10 \
		-n 1000 \
		-t 30 \
		-o calc/image_classification/distilbert_cifar10_n1000_t30.csv
	python src/text_models/bert.py \
		-d cifar10 \
		-n 10000 \
		-t 30 \
		-o calc/image_classification/distilbert_cifar10_n10000_t30.csv


resnet_tests:
	mkdir -p calc/image_classification
	python src/vision_models/resnet.py \
		-d fashion_mnist \
		-n 100 \
		-t 30 \
		-o calc/image_classification/resnet_fashionmnist_n100_t30.csv
	python src/vision_models/resnet.py \
		-d fashion_mnist \
		-n 1000 \
		-t 30 \
		-o calc/image_classification/resnet_fashionmnist_n1000_t30.csv
	python src/vision_models/resnet.py \
		-d fashion_mnist \
		-n 10000 \
		-t 30 \
		-o calc/image_classification/resnet_fashionmnist_n10000_t30.csv
	python src/vision_models/resnet.py \
		-d mnist \
		-n 100 \
		-t 30 \
		-o calc/image_classification/resnet_mnist_n100_t30.csv
	python src/vision_models/resnet.py \
		-d mnist \
		-n 1000 \
		-t 30 \
		-o calc/image_classification/resnet_mnist_n1000_t30.csv
	python src/vision_models/resnet.py \
		-d mnist \
		-n 10000 \
		-t 30 \
		-o calc/image_classification/resnet_mnist_n10000_t30.csv
	python src/vision_models/resnet.py \
		-d cifar10 \
		-n 100 \
		-t 30 \
		-o calc/image_classification/resnet_cifar10_n100_t30.csv
	python src/vision_models/resnet.py \
		-d cifar10 \
		-n 1000 \
		-t 30 \
		-o calc/image_classification/resnet_cifar10_n1000_t30.csv
	python src/vision_models/resnet.py \
		-d cifar10 \
		-n 10000 \
		-t 30 \
		-o calc/image_classification/resnet_cifar10_n10000_t30.csv


cvt_tests:
	mkdir -p calc/image_classification
	python src/vision_models/cvt.py \
		-d fashion_mnist \
		-n 100 \
		-t 30 \
		-o calc/image_classification/cvt_fashionmnist_n100_t30.csv
	python src/vision_models/cvt.py \
		-d fashion_mnist \
		-n 1000 \
		-t 30 \
		-o calc/image_classification/cvt_fashionmnist_n1000_t30.csv
	python src/vision_models/cvt.py \
		-d fashion_mnist \
		-n 10000 \
		-t 30 \
		-o calc/image_classification/cvt_fashionmnist_n10000_t30.csv
	python src/vision_models/cvt.py \
		-d mnist \
		-n 100 \
		-t 30 \
		-o calc/image_classification/cvt_mnist_n100_t30.csv
	python src/vision_models/cvt.py \
		-d mnist \
		-n 1000 \
		-t 30 \
		-o calc/image_classification/cvt_mnist_n1000_t30.csv
	python src/vision_models/cvt.py \
		-d mnist \
		-n 10000 \
		-t 30 \
		-o calc/image_classification/cvt_mnist_n10000_t30.csv
	python src/vision_models/cvt.py \
		-d cifar10 \
		-n 100 \
		-t 30 \
		-o calc/image_classification/cvt_cifar10_n100_t30.csv
	python src/vision_models/cvt.py \
		-d cifar10 \
		-n 1000 \
		-t 30 \
		-o calc/image_classification/cvt_cifar10_n1000_t30.csv
	python src/vision_models/cvt.py \
		-d cifar10 \
		-n 10000 \
		-t 30 \
		-o calc/image_classification/cvt_cifar10_n10000_t30.csv


all: bert_tests resnet_tests cvt_tests
