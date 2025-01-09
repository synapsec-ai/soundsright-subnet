FROM python:3.10.14-bookworm

# Copy required files
RUN mkdir -p /SoundsRightSubnet && mkdir -p /home/$USERNAME/.bittensor && mkdir -p /home/$USERNAME/.SoundsRightSubnet
COPY soundsright /SoundsRightSubnet/soundsright
COPY pyproject.toml /SoundsRightSubnet
COPY .env /SoundsRightSubnet

RUN /bin/bash -c "python3 -m venv /SoundsRightSubnet/.venv && source /SoundsRightSubnet/.venv/bin/activate && pip3 install pesq==0.0.4 && pip3 install -e /SoundsRightSubnet/.[validator]"