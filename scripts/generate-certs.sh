#!/usr/bin/env bash
# Generate a self-signed TLS certificate for local development.
# For production, replace nginx/certs/cert.pem and key.pem with real certificates.

set -euo pipefail

CERTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/nginx/certs"
mkdir -p "$CERTS_DIR"

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout "$CERTS_DIR/key.pem" \
  -out "$CERTS_DIR/cert.pem" \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

echo "Certificates written to $CERTS_DIR"
