file=$1
dir=$2
set -x
curl -upublic:Password@123 -T $file "http://10.10.1.53:8081/artifactory/model_zoo2/$dir/$file"
