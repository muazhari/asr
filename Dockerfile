# Initialize the base image.
FROM muazhari/research-assistant-backend:latest

# SSH server setup.
RUN apt update -y

RUN yes | DEBIAN_FRONTEND=noninteractive apt install -y \
    openssh-server

RUN mkdir /var/run/sshd

RUN echo 'root:root' | chpasswd

RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# Copy files to the working directory.
COPY . .

# Install pip dependencies.
RUN pip3 install -r requirements-dev.txt --use-feature=fast-deps --break-system-packages


