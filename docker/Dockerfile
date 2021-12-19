FROM registry.gitlab.com/nyker510/analysis-template/gpu:45e6cc9a5c4fb3a469b74bb197005c8536ab44b6

USER root
RUN pip install -U pip && \
  pip install \
  pandas==1.2.2 \
  pytorch-tabnet
  
USER penguin
WORKDIR /home/penguin

