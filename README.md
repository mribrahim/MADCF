# MADCF

Motion-Aware Object Tracking for Aerial Images with Deep Features and Discriminative Correlation Filter

[paper](https://link.springer.com/article/10.1007/s11042-024-18571-8)

> PESMOD and VIVID folder contains tracker annotations. You can use them to evaluate your trackers. 

> scripts/show-PESMOD.py and scripts/show-VIVID.py shows sequences int the datasets.

Download PESMOD dataset [from](https://github.com/mribrahim/PESMOD/)

# Build and Run

```
mkdir build
cd build
cmake ..
make
./tracker -p "/home/ibrahim/Desktop/Dataset/PESMOD/" -s Pexels-Shuraev-trekking
```


## Sample tracker outputs

UAV123L - car3: https://youtu.be/gz6GFC0MOm0

UAV123L - car9: https://youtu.be/ZX6RacGXPj4

UAV123L - group3: https://youtu.be/v5NkgOP__UY

UAV123L - person14: https://youtu.be/BgZSY1cm-QQ

UAV123L - person19: https://youtu.be/BT7vdlH1vYI

PESMOD: https://youtu.be/GuQeFZjCwy8
