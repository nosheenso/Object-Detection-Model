## YOLO Application Examples
This folder contains example code showing different applications you can build around YOLO models.

### Candy Calorie Counter
The [candy_calorie_counter](candy_calorie_counter) example uses a custom YOLO model that's trained to identify popular types of candy (Skittles, Snickers, etc). When candy is placed in front of the camera, the application checks the number of calories and grams of sugar contained in each piece of candy, and it reports the total calories and sugar. It's a basic example of how to use detected object classes to look up information about each object.

### Using YOLO With Multiple Cameras
The [multi_camera](multi_camera) example shows an efficient way to run YOLO models on multiple camera streams using Python multiprocessing.
