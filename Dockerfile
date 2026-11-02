# Tensorizer backend for a Hugging Face Docker Space (free CPU). The site itself stays on GitHub Pages and
# calls this API; see docs/deploy-huggingface.md.
FROM ghcr.io/prefix-dev/pixi:0.70.0-bookworm

# Hugging Face runs the container as user 1000.
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user pixi.toml pixi.lock ./
RUN pixi install --locked && rm -rf /root/.cache
COPY --chown=user . .
RUN chown -R user /app
USER user

ENV CORS_ORIGINS=https://jackyko1991.github.io \
    TENSORIZER_SHARED=1 \
    TENSORIZER_CACHE=/tmp/tensorizer-runs \
    TENSORIZER_MAX_CONCURRENT=1 \
    TENSORIZER_MAX_QUEUE=20 \
    TENSORIZER_PER_CLIENT=1 \
    AGENT_ENABLED=1 \
    MAX_AGENT_SESSIONS=2 \
    OMP_NUM_THREADS=2 \
    MPLBACKEND=Agg

EXPOSE 7860
CMD ["pixi", "run", "--locked", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
