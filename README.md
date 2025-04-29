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
### Train the model
```shell

pip install tensorflow mediapipe opencv-python scikit-learn
sudo apt install edgetpu-compiler


```

### Data

https://storage.googleapis.com/kaggle-data-sets/408863/783233/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250429%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250429T115923Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=114d4712506a6b3783e2d6fcf25a22166ac6a41649d9834d1de21cce58d3daac931f7ccb4e24786bd2513946442714c794b345f32400203765c67edafd29fd26b4f95e4d5a8a6ee1e1ce36dd2004cf319a98e55b2228e0b8886f6101bc60a216491516a943d877ac7dc37b0008ee2a007e56cf6ae90ecf766ba254665c4c6cd40d0d24aacb2e5c7a1eaad02b11bad199b57237d62fa453efa9ef83174ce13dbaa14c1b048a5048cc3293c1d52c0cb6a6afd12cd8601b4e2b284eaec2094a06b5cbf3639ad04333c9fb32e79aee4c841979230bdf67600d94e5a874c3c76b720695f9f918f25f0ef513173c816a83c113be83c224e00b80654299793a274535ba
https://storage.googleapis.com/mediapipe-tasks/object_detector/android_figurine.zip