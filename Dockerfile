FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ARG UID="11111"
ENV GID="0"
ARG UNAME=okaif

WORKDIR /app

# Install dependencies
RUN apt-get update \
  && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get update \
  && apt-get install -y --no-install-recommends python3.10 \
    dumb-init ca-certificates procps curl vim \
  && apt-get install -y libgl1

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

RUN mkdir -p /opt/okaif/etc/models/

RUN curl -L 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth' \
  -o /opt/okaif/etc/models/RealESRGAN_x4plus.pth
RUN chmod og+r /opt/okaif/etc/models/RealESRGAN_x4plus.pth

RUN curl -L 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth' \
  -o /opt/okaif/etc/models/GFPGANv1.4.pth
RUN chmod og+r /opt/okaif/etc/models/GFPGANv1.4.pth

RUN curl -L 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth' \
  -o /opt/okaif/etc/models/codeformer.pth
RUN chmod og+r /opt/okaif/etc/models/codeformer.pth

RUN mkdir -p /opt/okaif/bin/gfpgan/weights/

RUN curl -L 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth' \
  -o /opt/okaif/bin/gfpgan/weights/detection_Resnet50_Final.pth
RUN chmod og+r /opt/okaif/bin/gfpgan/weights/detection_Resnet50_Final.pth

RUN curl -L 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth' \
  -o /opt/okaif/bin/gfpgan/weights/parsing_parsenet.pth
RUN chmod og+r /opt/okaif/bin/gfpgan/weights/parsing_parsenet.pth

# Install Python dependencies
COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir --break-system-packages --default-timeout=100 -r requirements.txt \
    # Remove temporary files
    && rm -rf /root/.cache

RUN apt-get install sudo \
  && echo '%sudo ALL=(ALL:ALL) NOPASSWD:ALL' >/etc/sudoers.d/nopasswd \
  && groupadd -f -g ${GID} -o ${UNAME} \
  && useradd -m -u ${UID} -g ${GID} -o -s /bin/bash --create-home ${UNAME} \
  && usermod -a -G sudo ${UNAME} \
  && mkdir -p /opt/okaif/var/log/ \
  && chown ${UNAME} /opt/okaif/var/log/ # folder permissions can be changed after mounting !

COPY rootfs /
COPY src/bin/ /opt/okaif/bin/
COPY src/lib/ /opt/okaif/lib/

USER ${UID}

ENTRYPOINT ["/opt/okaif/bin/UpscaleServerRun.sh"]
