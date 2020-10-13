anpr - An Automatic Number-Plate Recognition module
=======================================================


An Automatic Number-Plate Recognition module using OpenCV & TensorFlow



Overview
--------

It's a simple module for Number-Plate Recognition that first uses OpenCV for detecting number plate and segmenting its characters then uses A Convolutional Neural Network for recognition of characters. The codes for training model is available in ``train_ocr_model.py``. You can use ``anpr.py`` easily as follow:

.. code-block:: bash

    $ ./anpr.py car.jpg
    number plate = B2228HM


Documentation
-------------

You can use verbose mode by adding -v (or --verbose) option to check process's steps:

.. code-block:: bash

    $ ./anpr.py car.jpg -v

- Step0 - Reading the image file

    .. image:: steps/step0-original.png
        :align: center
        :width: 300

- Step1 - RGB to gray scale conversion

    .. image:: steps/step1-grayscale.png
        :align: center
        :width: 300

- Step2 - Noise removal with iterative bilateral filter (removing noise while preserving edges)

    .. image:: steps/step2-bilateral.png
        :align: center
        :width: 300

- Step3 - Finding Edges by Canny edge detector

    .. image:: steps/step3-edged.png
        :align: center
        :width: 300

- Step4 - Extracting numberplate by find the best possible approximate contour 

    .. image:: steps/step4-img_plate.png
        :align: center
        :width: 300

- Step5 - RGB to gray scale conversion of number plate

    .. image:: steps/step5-gray_scale_plate.png
        :align: center
        :width: 300

- Step6 - Using thresholding method for binarizing number plate

    .. image:: steps/step6-binarize_plate.png
        :align: center
        :width: 300

- Step7 - Segmenting characters on number plate

    .. image:: steps/step7-roi_img0.png
        :align: center
        :width: 20
    .. image:: steps/step8-roi_img1.png
        :align: center
        :width: 20
    .. image:: steps/step9-roi_img2.png
        :align: center
        :width: 20
    .. image:: steps/step10-roi_img3.png
        :align: center
        :width: 20
    .. image:: steps/step11-roi_img4.png
        :align: center
        :width: 20
    .. image:: steps/step12-roi_img5.png
        :align: center
        :width: 20
    .. image:: steps/step13-roi_img6.png
        :align: center
        :width: 20

- Step8 - Ending detection and feeding segments to our CNN model

    .. image:: steps/step14-end_of_detection.png
        :align: center
        :width: 300

- Step9 - model output

    .. code-block:: bash

        number plate = B2228HM




Note
-------------

If you are unsuccessful in the recognizing number plate, Try to provide other optional arguments as well:

.. code-block:: bash

    $  ./anpr.py -h
    usage: anpr.py [-h] [-s S S] [-L L] [-v] file

    License Plate Detection and Recognition

    positional arguments:
        file                  image file

    optional arguments:
        -h, --help            show this help message and exit
        -s S S, --approx_size S S
                              approximate size of plate(the default value is (w/ 3, h / 19))
        -L L, --plate-length L
                            an integer for the accumulator(the default value of plate_lengthi is 7)
        -v, --verbose         save steps' images in current directory

License
-------

.. image:: https://img.shields.io/pypi/l/tfinance?color=green