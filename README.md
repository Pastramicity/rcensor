# rcensor

## Description

***Currently only works on Linux***

A lightweight eye censoring program. Shows a camera display which covers one figure's eyes with a black bar over both. The weight we are using for testing is linked [here](https://github.com/opencv/opencv/blob/4.x/data/lbpcascades/lbpcascade_frontalface_improved.xml).
This file should be placed in the `/opt/opencv/eye_detection.xml` directory on your machine.
This program also requires the installation of [opencv](https://opencv.org). 

## Installation

For this, you will need to install a few things onto your system, which are listed here:

1. opencv
2. clang
3. qt5-base
4. hdf5
5. cargo

Copy the xml weights for eye detection at [this link](https://raw.githubusercontent.com/opencv/opencv/4.x/data/lbpcascades/lbpcascade_frontalface_improved.xml) and place them at "/opt/opencv/eye_detection.xml"

Then clone this repository:
```bash
git clone "https://github.com/pastramicity/rcensor.git"
```

... `cd` into the repository:
```bash
cd rcensor
```

... and run it with:
```bash
cargo run
```
