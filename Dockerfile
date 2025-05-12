FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools wheel
RUN apt-get install git -y
RUN apt-get install tmux -y && echo "set -g mouse on" > ~/.tmux.conf

WORKDIR /amnFBA_workdir

COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME
