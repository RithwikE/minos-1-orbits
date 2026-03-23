FROM condaforge/miniforge3:24.11.3-0

WORKDIR /opt/minos-1-orbits

ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/mplconfig \
    XDG_CACHE_HOME=/tmp/xdg-cache

RUN mamba install -y -n base -c conda-forge \
    awscli \
    pykep \
    pygmo \
    pandas \
    scipy=1.13 \
    matplotlib \
    && mamba clean -afy \
    && mkdir -p /tmp/mplconfig /tmp/xdg-cache

COPY v2/shared ./v2/shared
COPY v2/01_joi ./v2/01_joi

ENTRYPOINT ["python", "v2/01_joi/run_container_job.py"]
