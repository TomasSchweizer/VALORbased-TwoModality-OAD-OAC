#!/bin/bash
conda init bash
localWorkspaceFolderBasename="$(dirname "$(dirname "$(readlink -fm "$0")")")"
"/opt/conda/bin/pip" install --user -r "$localWorkspaceFolderBasename/requirements.txt"

"/opt/conda/bin/pip" install --user mmengine
"/opt/conda/bin/pip" install --user mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

if [ ! -d "$localWorkspaceFolderBasename/mmaction2" ] ; then
    git clone https://github.com/open-mmlab/mmaction2.git
    export PYTHONPATH=$PYTHONPATH:$localWorkspaceFolderBasename/mmaction2:
    cd ./mmaction2
    "/opt/conda/bin/pip" install --user -v -e .
else
    export PYTHONPATH=$PYTHONPATH:$localWorkspaceFolderBasename/mmaction2
    cd ./mmaction2
    "/opt/conda/bin/pip" install --user -v -e .
    cd ..
fi
wandb login #your login code     
