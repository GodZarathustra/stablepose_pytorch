# BOP DATASET: T-LESS [1]


## Dataset parameters

* Objects: 30
* Object models:
    1) Reconstructed models with surface color and normals.
    2) CAD models with surface normals.
* Training images:
    1) 37584 real images (648 or 1296 per object) from each of the 3 sensors
    2) 76860 rendered images (2562 per object)
* Test images: 10080 real images from each of the 3 sensors
* Distribution of the ground truth poses in test images:
    * Range of object distances: 650 - 940 mm (from the Primesense camera)
    * Azimuth range: 0 - 360 deg
    * Elevation range: -90 - 90 deg


## Real and rendered training images

Besides real training images, the dataset includes rendered training images
obtained by rendering the reconstructed object models from a full densely
sampled view sphere with the radius of 650 mm. The intrinsic parameters of the
Primesense sensor were used for the rendering.


## Sensors

T-LESS includes images from three synchronized sensors:
1) Primesense CARMINE 1.09 (a structured-light RGB-D sensor)
2) Microsoft Kinect v2 (a time-of-flight RGB-D sensor)
3) Canon IXUS 950 IS (a high-resolution RGB camera)

Images from the Primesense sensor are used in the BOP paper [2] and provided in
the BOP Challenge 2019. Images from the other sensors can be downloaded from:
http://cmp.felk.cvut.cz/t-less


## Dataset format

The dataset format is described in:
https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md
https://github.com/thodan/t-less_toolkit/blob/master/doc/t-less_doc.md


## References

[1] Hodan et al., "T-LESS: An RGB-D Dataset for 6D Pose Estimation of
    Texture-less Objects", WACV 2017, web: http://cmp.felk.cvut.cz/t-less

[2] Hodan, Michel et al., "BOP: Benchmark for 6D Object Pose Estimation",
    ECCV 2018, web: http://bop.felk.cvut.cz
