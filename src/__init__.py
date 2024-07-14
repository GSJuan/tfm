"""
# When entering the machine: 

eval "$(ssh-agent -s)"
ssh-add .ssh/id_ed25519

pip install rdkit
pip install transformers-cfg
pip install numpy==1.24.2 # minimo necesario para q funque transformers-CFG y a la vez maximo para scipy (que lo usa bits and bytes)

# Para generar diagramas de clases del código
sudo add-apt-repository universe
sudo apt update
apt-get install graphviz
cd tfm
pyreverse -o png -p TMP src/


#Con las dos siguientes funca tb mixtral pero mas lento
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    
    pip install protobuf>=4.25.2 # para q funque transformers-cfg, necesario solo si has cambiado la version de protobuff despues de instalar transformers-cfg

#Con esto funciona solo mixtral sin transformers cfg pero a mejor rendimiento
pip install protobuf==3.20.1 # para q funque bit and bytes normal

pip install git+https://github.com/uiuc-focal-lab/syncode.git


IF using Syncode:
        nano /opt/conda/lib/python3.10/site-packages/syncode/common.py
            Change the HF_CACHE default value to None
        export HF_CACHE="/home/jovyan/.cache/huggingface"
        export SYNCODE_CACHE="/home/jovyan/.cache/syncode"

#########################################
"""