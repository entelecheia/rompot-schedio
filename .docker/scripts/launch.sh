#!/bin/bash
# add your custom commands here that should be executed every time the docker container starts

# print environment variables
export

echo "Starting docker container..."

### Set the $PUBLIC_KEY env var to enable SSH access.
# It is useful to have the full SSH server e.g. on Runpod.
# (use SCP to copy files to/from the image, etc)
if [[ -n "$SSH_PUB_KEY" ]] && [[ ! -d "${HOME}/.ssh" ]]; then
    mkdir -p "${HOME}/.ssh"
    echo "${SSH_PUB_KEY}" > "${HOME}/.ssh/authorized_keys"
fi
chmod -R 700 "${HOME}/.ssh"
sudo cp -f "$WORKSPACE_ROOT/scripts/sshd_config" "/etc/ssh/sshd_config"
sudo service ssh start

# start jupyter notebook in background and redirect output to logfile
# change working directory to workspace root
# set token to value of JUPYTER_TOKEN
# set port to value of JUPYTER_DOCKER_PORT
if [[ -z "$(command -v jupyter)" ]]; then
    echo "Jupyter not installed. Installing jupyter..."
    pip install jupyterlab
    echo "$PATH"
    # check if .local/bin is in $PATH, if not add it
    if ! echo "$PATH" | grep -q '.local/bin'; then
        echo "Adding .local/bin to PATH"
        export PATH="$PATH:$HOME/.local/bin"
    fi
fi
export PATH="$PATH:$HOME/.local/bin"
exec jupyter lab \
    --no-browser \
    --notebook-dir="$WORKSPACE_ROOT" \
    --ServerApp.token="$JUPYTER_TOKEN" \
    --port="$JUPYTER_PORT" \
    --ip=0.0.0.0 \
    --allow-root
