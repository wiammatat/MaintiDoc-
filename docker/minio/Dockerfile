# MinIO compilé depuis son code source officiel (github.com/minio/minio, licence AGPL 3.0).
# MinIO ne distribue plus d'images Docker ni de binaires publics : seul le code source reste disponible.
# Version figée pour la reproductibilité.

FROM golang:1.24.8-alpine3.22 AS build
ARG MINIO_RELEASE=RELEASE.2025-10-15T17-29-55Z
RUN apk add --no-cache git
RUN git clone --depth 1 --branch ${MINIO_RELEASE} https://github.com/minio/minio.git /src
WORKDIR /src
# Inscrit la version dans le binaire avec le script officiel de MinIO
# (RELEASE.2025-10-15T17-29-55Z -> 2025-10-15T17:29:55Z).
RUN VERSION=$(echo "${MINIO_RELEASE#RELEASE.}" | sed -E 's/T([0-9]{2})-([0-9]{2})-([0-9]{2})Z/T\1:\2:\3Z/') \
 && CGO_ENABLED=0 go build -trimpath -ldflags "-s -w $(go run buildscripts/gen-ldflags.go "$VERSION")" -o /out/minio .

FROM alpine:3.22.2
RUN apk add --no-cache ca-certificates curl && adduser -D -u 1000 minio
COPY --from=build /out/minio /usr/bin/minio
USER minio
EXPOSE 9000 9001
ENTRYPOINT ["minio"]
CMD ["server", "/data", "--console-address", ":9001"]
