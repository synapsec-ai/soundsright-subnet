import os
import tempfile
import pytest

import soundsright.base.utils as Utils

def create_temp_dockerfile(content):
    """Helper to create a temporary Dockerfile with given content."""
    temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    temp_file.write(content)
    temp_file.close()
    return temp_file.name

@pytest.mark.parametrize("content, expected", [
    ("FROM alpine\nUSER root", True),
    ("FROM ubuntu\nUSER 0", True),
    ("FROM alpine\nRUN echo hello", True),
    ("FROM debian\nUSER nobody", False),
    ("FROM ubuntu\nARG USER_ID=root\nUSER $USER_ID", True),
    ("FROM ubuntu\nARG USER_ID=nobody\nUSER $USER_ID", False),
    ("FROM alpine\nENV USER_ID=root\nUSER $USER_ID", True),
    ("FROM alpine\nENV USER_ID=nobody\nUSER $USER_ID", False),
    ("FROM alpine\nUSER $UNDEFINED", True),
    ("FROM busybox\nARG BADARG\nUSER root", True),
    ("""
    FROM python:3.10.14-bookworm

    ARG USER_UID=10002
    ARG USER_GID=$USER_UID
    ARG USERNAME=modelapi

    RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

    # Copy required files
    RUN mkdir -p /modelapi && mkdir -p /home/$USERNAME/.modelapi
    COPY app /modelapi/app
    COPY sgmse /modelapi/sgmse
    COPY pyproject.toml /modelapi/pyproject.toml

    ENV CUDA_HOME=/usr/local/cuda-12.6

    # Setup permissions
    RUN chown -R $USER_UID:$USER_GID /modelapi \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME/.modelapi \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME \
    && chmod -R 755 /home/$USERNAME \
    && chmod -R 755 /modelapi \
    && chmod -R 755 /home/$USERNAME/.modelapi

    # Change to the user and do subnet installation
    USER $USERNAME

    RUN /bin/bash -c "python3 -m venv /modelapi/.venv && source /modelapi/.venv/bin/activate && pip3 install -e /modelapi/."

    EXPOSE 6500

    CMD ["/bin/bash", "-c", "source /modelapi/.venv/bin/activate && python3 /modelapi/app/run.py"]

    """, False),
])
def test_dockerfile_root_detection(content, expected):
    path = create_temp_dockerfile(content)
    try:
        assert Utils.check_dockerfile_for_root_user(path) == expected
    finally:
        os.remove(path)

@pytest.mark.parametrize("content,expected", [
    ("""
    FROM ubuntu:20.04
    VOLUME [\"/root/.bittensor\"]
    """, True),
    ("""
    FROM python:3.9
    COPY . /app
    """, False),
    ("""
    VOLUME [\"/etc/config\", \"/var/log\"]
    """, True),
    ("", False),
    ("""
    FROM python:3.10.14-bookworm

    ARG USER_UID=10002
    ARG USER_GID=$USER_UID
    ARG USERNAME=modelapi

    RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

    # Copy required files
    RUN mkdir -p /modelapi && mkdir -p /home/$USERNAME/.modelapi
    COPY app /modelapi/app
    COPY sgmse /modelapi/sgmse
    COPY pyproject.toml /modelapi/pyproject.toml

    ENV CUDA_HOME=/usr/local/cuda-12.6

    # Setup permissions
    RUN chown -R $USER_UID:$USER_GID /modelapi \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME/.modelapi \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME \
    && chmod -R 755 /home/$USERNAME \
    && chmod -R 755 /modelapi \
    && chmod -R 755 /home/$USERNAME/.modelapi

    # Change to the user and do subnet installation
    USER $USERNAME

    RUN /bin/bash -c "python3 -m venv /modelapi/.venv && source /modelapi/.venv/bin/activate && pip3 install -e /modelapi/."

    EXPOSE 6500

    CMD ["/bin/bash", "-c", "source /modelapi/.venv/bin/activate && python3 /modelapi/app/run.py"]

    """, False),
])
def test_check_dockerfile_for_sensitive_config(content, expected):
    path = create_temp_dockerfile(content)
    try:
        assert Utils.heck_dockerfile_for_sensitive_config(str(path)) is expected
    finally:
        os.remove(path)
