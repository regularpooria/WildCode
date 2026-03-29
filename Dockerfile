FROM node:20-bookworm-slim

# Install OpenGrep and runtime prerequisites.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir opengrep

WORKDIR /app

# Copy only what the scanner server needs.
COPY github_pages_code_checking ./github_pages_code_checking
COPY opengrep-rules ./opengrep-rules

ENV NODE_ENV=production
ENV HOST=0.0.0.0
ENV PORT=10000
ENV OPENGREP_BIN=opengrep

EXPOSE 10000

CMD ["node", "github_pages_code_checking/server.js"]
