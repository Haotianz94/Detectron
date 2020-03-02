# Thanks @dustinfreeman for providing the script
#!/bin/bash
# nvidia-docker build -f docker/Dockerfile -t detect_and_track:CUDA9-py2 . 

# run from ~/Projects
nvidia-docker run -ti --ipc=host --shm-size 12G -v $(pwd):/Detectron --workdir=/Detectron haotianz/detect_and_track:CUDA9-py2 /bin/bash

# docker login

# docker tag detect_and_track:CUDA9-py2 docker.io/haotianz/detect_and_track:CUDA9-py2

# docker push haotianz/detect_and_track:CUDA9-py2

