# OpenShell

Run OAB inside an [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) sandbox for isolated, policy-enforced execution with credential injection.

## Prerequisites

- Docker running on the host
- [OpenShell CLI](https://github.com/NVIDIA/OpenShell#install) installed

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
```

## Quick Start (Local Docker)

The following is a single copy-pasteable sequence. Run all commands on the host unless noted otherwise.

```bash
# 1. Create credential providers
#    Providers are stored in the OpenShell gateway's local state.
#    Host env vars are read only at creation time and not retained.
#    Providers persist until explicitly removed with `openshell provider delete <name>`.
export DISCORD_BOT_TOKEN="your-token"
export GITHUB_TOKEN="your-token"
export ANTHROPIC_API_KEY="your-key"

openshell provider create --name discord --env DISCORD_BOT_TOKEN
openshell provider create --name github --env GITHUB_TOKEN
openshell provider create --name anthropic --env ANTHROPIC_API_KEY

# 2. Create sandbox with providers and port forwarding
openshell sandbox create --name oab \
  --provider discord \
  --provider github \
  --provider anthropic \
  --forward 3000 \
  -- bash

# 3. Apply network policy (all unlisted egress is denied by default)
cat > /tmp/oab-policy.yaml <<'EOF'
network:
  egress:
    - destination: "discord.com"
      ports: [443]
    - destination: "gateway.discord.gg"
      ports: [443]
    - destination: "api.github.com"
      ports: [443]
    - destination: "github.com"
      ports: [443]
    - destination: "api.anthropic.com"
      ports: [443]
EOF
openshell policy set oab --policy /tmp/oab-policy.yaml --wait

# 4. Connect to the sandbox
openshell sandbox connect oab
```

Inside the sandbox:

```bash
git clone https://github.com/openabdev/openab.git
cd openab
cargo build --release
./target/release/openab serve --config config.toml
```

At this point `localhost:3000` on the host reaches port 3000 inside the sandbox (useful for GitHub webhook delivery via grok/ngrok).

## Credential Management

| Operation | Command |
|-----------|---------|
| List providers | `openshell provider list` |
| Delete a provider | `openshell provider delete discord` |
| Rotate a credential | Delete + recreate with new value |

Credentials are injected as env vars at sandbox runtime. They are **not** written to the sandbox filesystem. Removing a provider immediately revokes access on the next sandbox restart.

## Port Forwarding

Add `--forward <port>` at sandbox creation. Multiple ports are supported:

```bash
openshell sandbox create --name oab \
  --provider discord \
  --forward 3000 \
  --forward 8080 \
  -- bash
```

Each forwarded port creates an SSH tunnel: `localhost:<port>` on the host → `127.0.0.1:<port>` inside the sandbox. Tunnels are torn down when the sandbox is deleted.

## BYOC (Custom Image)

Build a custom sandbox image with OAB pre-installed:

```dockerfile
FROM ubuntu:24.04

RUN groupadd -g 1000660000 sandbox && \
    useradd -u 1000660000 -g sandbox -m sandbox

RUN apt-get update && apt-get install -y \
    curl git iproute2 ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    su sandbox -c 'sh -s -- -y'

# Pre-clone and build OAB
USER sandbox
WORKDIR /home/sandbox
RUN . /home/sandbox/.cargo/env && \
    git clone https://github.com/openabdev/openab.git && \
    cd openab && cargo build --release

WORKDIR /home/sandbox/openab
```

Run it:

```bash
openshell sandbox create --name oab \
  --from ./Dockerfile \
  --provider discord \
  --provider github \
  --provider anthropic \
  --forward 3000 \
  -- bash

openshell policy set oab --policy /tmp/oab-policy.yaml --wait
openshell sandbox connect oab
```

Inside the sandbox, OAB is already built:

```bash
./target/release/openab serve --config config.toml
```

## Cleanup

```bash
openshell sandbox delete oab
# Optionally remove providers
openshell provider delete discord
openshell provider delete github
openshell provider delete anthropic
```
