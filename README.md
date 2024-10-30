# Beer Pong Trajectory Analysis

I am tired of constantly missing the last cup in a game of beer pong, that last cup is the bane of my existence. Thus, I decided I needed a training method to perfect my shot, what better way to do that then with computer visualization? This is an in progress project where the end goal is to create a sideview shot predictor for a pong ball going into a classic red-solo cup. 

This project is currently being written in python with the utilization of openCV and YOLOv11 newest object detention model. At its current state it can predict path or travel for a white pong ball, along with simultaneously using YOLOvll pretrained model to recognize cups.
## Next Steps
Coming in the future, I will be training the Yolo Model to recognize Red-Solo cups specifically, and recognize it as a goal status for the pong ball

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install openCV, and ultralytics.

```bash
pip install ultralytics
```

```bash
pip install openCV
```

## Usage

Once you have installed all the required dependencies, simply run the program in the IDE of your choice, if there is a problem with your webcam not displaying, double check the value for your computer's/desired webcams. Most laptops have a webcam value of 0. Change this value in the necessary place to access the correct camera and open a frame. 


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)