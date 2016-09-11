#!/usr/bin/env bash

dir=/home/jeremy/binary_images
for d in $dir/ ;
    do echo $d
    cd $d
    for f in *.jpg ;
        do echo $f;
        echo $d_$f;
        done;
    done