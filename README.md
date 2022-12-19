# Road Lanes Detection and Turn Prediction using Sliding Window Algorithm

This project is a student level project for the course "Perception for Autonomous Robots". It is an implementation of various concepts such as gaussian blurring, image warping, sliding window, and edge detection using the libraries opencv and numpy applied to a real time problem, to detect road lanes and predict upcoming turns.

## Project Description

The goal of this project was to develop a system that can detect road lanes and predict turns in real-time using a sliding window algorithm. To achieve this, we implemented the following steps:

1. **Preprocessing**: I applied gaussian blurring to reduce noise and image warping to correct for perspective distortion.
2. **Edge Detection**: I used edge detection techniques to identify the edges in the preprocessed image.
3. **Sliding Window**: I used a sliding window algorithm to identify the lane lines in the image by searching for the highest concentration of edges within each window.
4. **Turn Prediction**: I used the positions of the detected lane lines to predict the direction of the turn.

### Objective

The challenge is to not use any inbuilt functions from open source libraries but to implement the mathematics behind the functions and understand how things work behind the API. Once you have implemented both of them compare the result of each technique and validate them with the original dataset.

## Dependencies

To run this project, you will need to install all/some of the following:

- [OpenCV](https://docs.opencv.org/) `pip install opencv-python`
- [Numpy](https://numpy.org/doc/stable/) `pip install numpy`
- glob `pip install glob2`
- Download and install Anaconda {easy}

## Contents

```
├───LICENSE
├───proj2_part2_report.pdf
├───README.md
├───Lane_Detection.py
├───Turn_Prediction.py
├───challenge.mp4
├───whiteline.mp4
├───outputs prob 2
└───outputs prob 3
```

## Instructions to run

1. Clone the repository

   ```
   git clone https://github.com/bharadwaj-chukkala/Road-Lanes-detection-and-Turn-Prediction-using-Sliding-Window-Algorithm.git
   ```
2. Open the folder in the IDE
3. Lane Detection

   * Uses the ``whiteline.mp4`` video data
   * Run the `Lane_Detection.py` file
   * Uncomment the commented lines at the end to save outputs to outputs folder
4. Turn Prediction

   * Uses the `challenge.mp4` video data
   * Run the `Turn_Prediction.py` file
   * Uncomment the commented lines at the end to save outputs to outputs folder

## Results

### Lane Detection

<div class="row">
  <div class="column">
    <img width="300" height="150" src="https://github.com/bharadwaj-chukkala/Road-Lanes-detection-and-Turn-Prediction-using-Sliding-Window-Algorithm/blob/master/outputs%20prob%202/edge_image.jpg">
  </div>
  <div class="column">
    <img width="300" height="150" src="https://github.com/bharadwaj-chukkala/Road-Lanes-detection-and-Turn-Prediction-using-Sliding-Window-Algorithm/blob/master/outputs%20prob%202/masked_image.jpg">
  </div>
  <div class="column">
    <img width="300" height="150" src="https://github.com/bharadwaj-chukkala/Road-Lanes-detection-and-Turn-Prediction-using-Sliding-Window-Algorithm/blob/master/outputs%20prob%202/hough_image.jpg">
  </div>
</div>



Fig 1: Edge Detection

<img width="300" height="150" src="https://github.com/bharadwaj-chukkala/Road-Lanes-detection-and-Turn-Prediction-using-Sliding-Window-Algorithm/blob/master/outputs%20prob%202/edge_image.jpg">




## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
