## Make own Model

### Train the model

```shell
unzip assets/rps_data_sample.zip

docker build -t rps-tpu-builder .

docker run --rm \
  -v "$PWD/rps_data_sample":/app/training_data:ro \
  -v "$PWD/result":/app/build \
  rps-tpu-builder
```
