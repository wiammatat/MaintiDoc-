#!/bin/sh
# Génère la configuration d'identités S3 à partir des variables d'environnement,
# pour ne jamais écrire le secret dans un fichier versionné.
set -eu
printf '{"identities":[{"name":"maintidoc","credentials":[{"accessKey":"%s","secretKey":"%s"}],"actions":["Admin","Read","Write","List","Tagging"]}]}' \
  "$S3_ACCESS_KEY" "$S3_SECRET_KEY" > /tmp/s3.json
exec weed server -s3 -s3.config=/tmp/s3.json -dir=/data
