#!/bin/bash

# Ensure gdown is available
pip install gdown

##################################### Download FLAME related data #####################################
DATA_DIR=data

urle () { [[ "${1}"  ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++  )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-]  ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

echo -e "\nBefore you continue, you must register at https://flame.is.tue.mpg.de/ and agree to the FLAME license terms."
read -p "Username (FLAME):" username
read -p "Password (FLAME):" password
username=$(urle $username)
password=$(urle $password)

echo -e "\nDownloading generic_model.pkl ..."

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' -O "${DATA_DIR}/FLAME2020.zip" --no-check-certificate --continue
unzip ${DATA_DIR}/FLAME2020.zip -d ${DATA_DIR}/FLAME2020
mv ${DATA_DIR}/FLAME2020/generic_model.pkl ${DATA_DIR}
rm -rf ${DATA_DIR}/FLAME2020 && rm ${DATA_DIR}/FLAME2020.zip
# MD5: cfc3b95159b2a8981ebc347170202ba5  generic_model.pkl

echo -e "\nDownloading FLAME_texture.npz  ..."

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=TextureSpace.zip&resume=1' -O "${DATA_DIR}/TextureSpace.zip" --no-check-certificate --continue
unzip ${DATA_DIR}/TextureSpace.zip -d ${DATA_DIR}/TextureSpace
mv ${DATA_DIR}/TextureSpace/FLAME_texture.npz ${DATA_DIR}
rm -rf ${DATA_DIR}/TextureSpace && rm ${DATA_DIR}/TextureSpace.zip
# MD5: 2a3982db7667c32d0dd13351aa6758f7  FLAME_texture.npz

echo -e "\nDownloading deca_model..."

FILEID=1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje
FILENAME=${DATA_DIR}/deca_model.tar
gdown https://drive.google.com/uc\?id\=$FILEID -O $FILENAME
# MD5: b182e51c26e5a73b1f7a5c53cf8cc038  deca_model.tar

echo -e "\nDownloading others"

git clone https://github.com/yfeng95/DECA ${DATA_DIR}/DECA
mv ${DATA_DIR}/DECA/data/fixed_displacement_256.npy ${DATA_DIR}/fixed_displacement_256.npy
mv ${DATA_DIR}/DECA/data/head_template.obj ${DATA_DIR}/head_template.obj
mv ${DATA_DIR}/DECA/data/landmark_embedding.npy ${DATA_DIR}/landmark_embedding.npy
mv ${DATA_DIR}/DECA/data/mean_texture.jpg ${DATA_DIR}/mean_texture.jpg
mv ${DATA_DIR}/DECA/data/texture_data_256.npy ${DATA_DIR}/texture_data_256.npy
mv ${DATA_DIR}/DECA/data/uv_face_eye_mask.png ${DATA_DIR}/uv_face_eye_mask.png
mv ${DATA_DIR}/DECA/data/uv_face_mask.png ${DATA_DIR}/uv_face_mask.png
rm -rf ${DATA_DIR}/DECA
# MD5: d801f9faebdee1470f87c121193c730f  fixed_displacement_256.npy
# MD5: ddc6d31590f871ebe1844b063b2f8042  head_template.obj
# MD5: a275dadfa4c1bd3c71a9f1411d68f92c  landmark_embedding.npy
# MD5: 8a20c857028cfdd8b3ee5eec1ada341f  mean_texture.jpg
# MD5: 9e44b1640e8cd981f2704e1c206cda3b  texture_data_256.npy
# MD5: 513f1360be7529b72e3cefc7fdc0b506  uv_face_eye_mask.png
# MD5: a376ad16d34c7216dc1f8e183712c641  uv_face_mask.png

##################################### Download models #####################################
MODEL_DIR=ckpts

echo -e "\nDownloading the face mask parser model"

mkdir -p ${MODEL_DIR}/face-parsing
FILEID=154JgKpzCPW82qINcVieuPH3fZ2e0P812
FILENAME=${MODEL_DIR}/face-parsing/79999_iter.pth
gdown https://drive.google.com/uc\?id\=$FILEID -O $FILENAME
# MD5: ff26a222ce48a618a1fa820b46223cae  79999_iter.pth

echo -e "\nDownloading the vae model"

wget -c https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt -O ${MODEL_DIR}/vae-ft-mse-840000-ema-pruned.ckpt
# MD5: 984367df26caf6707a4ef7aa354d4df6  vae-ft-mse-840000-ema-pruned.ckpt

echo -e "\nDownloading the stable diffusion backbone model"

wget -c https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE/resolve/main/Realistic_Vision_V3.0.ckpt -O ${MODEL_DIR}/Realistic_Vision_V3.0.ckpt
# MD5: 24913fbd1ed9cb7500ee90191bc5d4b2  Realistic_Vision_V3.0.ckpt

echo -e "\nDownloading the our caphuman model"

wget -c https://huggingface.co/VamosC/CapHuman/resolve/main/caphuman.ckpt -O ${MODEL_DIR}/caphuman.ckpt
# MD5: 6f0086112db9abc6ce4518a458d4d4b8  caphuman.ckpt
