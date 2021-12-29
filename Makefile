OUTPUT_LOC ?= ./output

IMAGE_TAG = public.ecr.aws/o4s5x0l8/cut-detection
VERSION = latest

.PHONY: build ecr-login push pull cut-video

ecr-login: 
	aws ecr-public get-login-password --region us-east-1 | sudo docker login --username AWS --password-stdin public.ecr.aws/o4s5x0l8

build: Dockerfile
	docker build -t $(IMAGE_TAG):$(VERSION) -f Dockerfile .

push: ecr-login
	docker push $(IMAGE_TAG):$(VERSION)

pull:
	docker pull $(IMAGE_TAG):$(VERSION)

cut-video: pull
	docker run -v $(OUTPUT_LOC):./sources --gpus all $(IMAGE_TAG):$(VERSION) $(TAPE)

cut-video-cpu: pull
	docker run -v $(OUTPUT_LOC):./sources $(IMAGE_TAG):$(VERSION) $(TAPE)
	