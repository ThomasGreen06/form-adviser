docker-build:
	docker image build -t form-adviser-flask .
	docker image ls

docker-container:
	docker run --rm -d -p 5000:5000 --name form-adviser-flask form-adviser-flask