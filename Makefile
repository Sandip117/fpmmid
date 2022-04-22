### fpmmid build and installation ###
.PHONY: config install check clean build
root_dir:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

config:
	@echo "----Creating fpmmid virtual environment and installing compiler----"
	python3 -m venv $(root_dir)/fpmmid-env
	$(root_dir)/fpmmid-env/bin/pip3 install --upgrade pip==21.3.1
	$(root_dir)/fpmmid-env/bin/pip3 install -r $(root_dir)/src/cmake.txt

build:
	@echo "----install horovod, tensorflow, pyspark, and post-processing modules----"
	export PATH=$(root_dir)/fpmmid-env/bin:$(PATH) && $(root_dir)/fpmmid-env/bin/pip3 install -r $(root_dir)/src/requirements.txt

check:
	@echo "----check horovod's build----"
	$(root_dir)/fpmmid-env/bin/horovodrun --check-build

clean:
	@echo "----Removing fpmmid virtual environment----"
	rm -rf $(root_dir)/fpmmid-env

install: config build check

all: clean config build check
