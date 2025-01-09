FROM python:3.10.14-bookworm

# Copy required files
RUN mkdir -p /soundsright-subnet && mkdir -p /home/$USERNAME/.bittensor && mkdir -p /home/$USERNAME/.soundsright-subnet
COPY soundsright /soundsright-subnet/soundsright
COPY pyproject.toml /soundsright-subnet
COPY .env /soundsright-subnet

RUN /bin/bash -c "python3 -m venv /soundsright-subnet/.venv && source /soundsright-subnet/.venv/bin/activate && pip3 install pesq==0.0.4 && pip3 install -e /soundsright-subnet/.[validator]"