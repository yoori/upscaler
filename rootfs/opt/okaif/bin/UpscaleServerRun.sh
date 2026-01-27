#!/bin/bash

sudo -E bash -c "j2 --import-env= /opt/okaif/etc/upscale_server.conf.templ >/opt/okaif/etc/upscale_server.conf"

# Part of mounted volume premissions workaround
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)
sudo -n chown "$CURRENT_UID:$CURRENT_GID" /opt/okaif/var/log

mkdir -p /opt/okaif/var/log/upscaler/ # create log folder if it isn't mounted

export PYTHONPATH=$PYTHONPATH:/opt/okaif/lib/

# gfpgan use model weights relative to current dir (in gfpgan/weights/) ...
cd /opt/okaif/bin/

CONFIG_PATH=/opt/okaif/etc/upscale_server.conf \
  gunicorn -b 0.0.0.0:8080 \
    --worker-class uvicorn.workers.UvicornWorker upscale_server:app \
  >/opt/okaif/var/log/upscaler/upscaler.log 2>&1
