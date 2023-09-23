from nvcr.io/nvidia/pytorch:22.12-py3
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
# Please choose previous pytorch:xx.xx if you encounter cuda driver mismatch issue
RUN pip install --upgrade pip
RUN pip install torchaudio==0.13.1
RUN pip3 install \
    k2==1.23.3.dev20230105+cuda11.7.torch1.13.1 -f https://k2-fsa.org/nightly/
# #install k2 from source
#"sed -i ..."  line tries to turn off the cuda check
RUN git clone https://github.com/k2-fsa/k2.git && \
    cd k2 && \
    sed -i 's/FATAL_ERROR/STATUS/g' cmake/torch.cmake && \
    sed -i 's/in running_cuda_version//g' get_version.py && \
    python3 setup.py install && \
    cd -
WORKDIR /workspace

RUN git clone https://github.com/k2-fsa/icefall.git
ENV PYTHONPATH "${PYTHONPATH}:/workspace/icefall"
# https://github.com/k2-fsa/icefall/issues/674
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION "python"

WORKDIR /workspace
RUN git clone https://github.com/lifeiteng/valle.git
WORKDIR /workspace/valle 
RUN pip install -e .
