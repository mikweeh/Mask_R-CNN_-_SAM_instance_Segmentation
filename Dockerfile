# Use any official python image
FROM superlinear/python-gpu:3.10-cuda11.8

# Keep Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turn off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Working directory. ws stands for workspace
WORKDIR /ws

# Install system dependencies (OpenCV, Tkinter, Git)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    libtcl8.6 libtk8.6 tk \
    git \
    x11-apps \
    xauth \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip requirements
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/facebookresearch/sam2.git

# Copy source code
COPY ./src ./src

# Define arguments for user and group IDs
ARG UID=1000
ARG GID=1000

# Create a non-root user with the specified UID and GID, handling existing groups
RUN if getent group $GID >/dev/null; then \
    existing_group=$(getent group $GID | cut -d: -f1); \
    useradd -m -u $UID -g $existing_group -s /bin/bash appuser; \
else \
    groupadd -g $GID appuser && \
    useradd -m -u $UID -g $GID -s /bin/bash appuser; \
fi

# Set ownership and permissions
RUN chown -R $UID:$GID /ws

# Switch to non-root user
USER appuser

# Expose the debugging port (5678)
EXPOSE 5678

# Expose other necessary ports
# EXPOSE 80

# Execute your application
CMD ["python", "src/main.py"]
