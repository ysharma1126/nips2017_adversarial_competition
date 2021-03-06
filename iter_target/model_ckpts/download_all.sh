#!/usr/bin/env bash

# Inception V3
if [ ! -f inception_v3.ckpt ]; then
    wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    tar -xvf inception_v3_2016_08_28.tar.gz
    rm inception_v3_2016_08_28.tar.gz
fi

if [ ! -f adv_inception_v3.ckpt ]; then
    wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
    tar -xvf adv_inception_v3_2017_08_18.tar.gz
    rm adv_inception_v3_2017_08_18.tar.gz
fi

if [ ! -f ens_adv_inception_resnet_v2.ckpt ]; then
    wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    tar -xvf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz
fi



