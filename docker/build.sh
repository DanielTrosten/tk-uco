echo "Dockerhub username: "
read username

cp requirements.txt docker/tmp_req.txt

docker build \
  --tag $username/tensor-kernel-clustering \
  --rm \
  --no-cache \
  docker/

rm docker/tmp_req.txt
