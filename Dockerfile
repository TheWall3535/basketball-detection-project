# 1. Start with an official Jupyter base image.
FROM jupyter/base-notebook:latest

# 2. Switch to the root user to install system-level packages.
USER root

# 3. Update package lists and install essential libraries for PyTorch and computer vision.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 4. Switch back to the default non-root user.
USER jovyan

# 5. Set the working directory.
WORKDIR /home/jovyan/work

RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# 6. Copy your local files into the container.
COPY requirements.txt ./

# 7. Install the Python packages.
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# 8. Expose port 8001.
EXPOSE 8001

# 9. Define the command to run when the container starts.
CMD ["jupyter", "notebook", "--port=8001", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.token=''"]
