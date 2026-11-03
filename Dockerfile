
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# ---- OS deps for xTB ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates tar bash coreutils gnupg xz-utils \
 && rm -rf /var/lib/apt/lists/*

# ---- Install xTB ----
ENV XTB_VERSION=6.7.1

RUN wget -q "https://github.com/grimme-lab/xtb/releases/download/v${XTB_VERSION}/xtb-${XTB_VERSION}-linux-x86_64.tar.xz" \
 && tar -xJf xtb-${XTB_VERSION}-linux-x86_64.tar.xz \
 && rm xtb-${XTB_VERSION}-linux-x86_64.tar.xz \
 && XTBDIR=$(find . -maxdepth 1 -type d -name "xtb*") \
 && mv "$XTBDIR" /opt/xtb \
 && ln -s /opt/xtb/bin/xtb /usr/local/bin/xtb

ENV XTBHOME=/opt/xtb/share/xtb
ENV PATH=/opt/xtb/bin:${PATH}
ENV OMP_STACKSIZE=4G
ENV OMP_NUM_THREADS=4

# ---- Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ---- Project code ----

# Copy data folder (contains anion-external-data.csv and possibly models/)
COPY data ./data

# Create /app/models and copy the model if it exists
RUN mkdir -p /app/models && \
    if [ -f data/models/model.joblib ]; then \
        cp data/models/model.joblib /app/models/model.joblib; \
    fi

# Copy Python packages
COPY pipeline ./pipeline
COPY complexes ./complexes
COPY constants ./constants
COPY interactions ./interactions
COPY molecular_structure ./molecular_structure

COPY tests ./tests

# Ensure calculate_input_affinity.py is copied
COPY pipeline/calculate_input_affinity.py ./pipeline/calculate_input_affinity.py



# ---- Entry point ----
ENTRYPOINT ["python", "pipeline/run_pipeline.py"]
