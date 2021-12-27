OUTPUT_LOC ?= ./output

IMAGE_TAG = play4honor/cut-detector
VERSION = latest

.PHONY: build push pull cut-video

build: Dockerfile
	docker build -t $(IMAGE_TAG):$(VERSION) -f Dockerfile .

push:
	docker push $(IMAGE_TAG):$(VERSION)

pull:
	docker pull $(IMAGE_TAG):$(VERSION)

cut-video: pull
	docker run -v $(OUTPUT_LOC):./output $(IMAGE_TAG):$(VERSION) $(TAPE)
