Social Distance Monitoring Using MobileNet-SSD

This project implements a real-time social distancing and object detection system using the MobileNet-SSD deep learning model with OpenCV. It detects people from video streams, tracks their positions, and highlights cases where two or more individuals are standing too close to each other based on distance thresholds.

The repository includes pre-trained MobileNet-SSD Caffe models, Python scripts, and multiple sample videos for testing.

🚀 Features

✔ Real-time object detection using MobileNet-SSD

✔ Detection of people, dogs, pedestrians, and other common COCO classes

✔ Social-distance calculation using bounding box centroids

✔ Visual alerts for distance violations

✔ Works on video files or live camera feed

✔ Lightweight and fast — runs on CPU

📂 Project Structure
├── social_distancing.py          # Main script with tracking + distance measurement
├── MobileNetSSD_deploy.caffemodel
├── MobileNetSSD_deploy.prototxt
├── SampleOne.mp4
├── public-video.mp4
├── public-video2.mp4
├── pedestrians.mp4
├── dogs-video.mp4
├── Create.mp4
├── School_Bag_Stock_Footage_Royalty_Free_Video.mp4
├── __pycache__/
└── .gitattributes

🧠 How It Works

The script loads the MobileNet-SSD deep learning model.

Each frame of the video is processed to detect objects.

Only the class "person" is used for social-distance estimation.

For every pair of detected individuals:

Their centroid distance is computed.

If the distance is below a threshold (in pixels), the pair is flagged.

Bounding boxes are color-coded:

🟢 Safe distance

🔴 Violation (too close)

🛠️ Installation
1. Clone the repository
git clone https://github.com/yourusername/social-distance-monitoring-using-mobilenet-ssd.git
cd social-distance-monitoring-using-mobilenet-ssd

2. Install dependencies
pip install opencv-python imutils numpy

▶️ Usage
Run on a sample video
python social_distancing.py --video public-video.mp4

Run on webcam
python social_distancing.py --camera 0

Optional arguments
--video <path>      Path to input video file
--camera <id>       Webcam device index (0 = default camera)
--threshold <val>   Distance threshold for violations

🧪 Sample Videos Included

The repo contains multiple videos for testing:

Public crowd videos

Pedestrian footage

Dog detection demo

School bag royalty-free footage

Create.mp4 (custom testing video)

These help evaluate different environments and crowd densities.

📦 Model Information

This project uses the MobileNet-SSD V1 Caffe model:

MobileNetSSD_deploy.prototxt

MobileNetSSD_deploy.caffemodel

Classes supported include: person, dog, bicycle, car, chair, bottle, and many more.

📸 Output Examples

Detected frames will show:

Bounding boxes around people

Centroid points

Lines between individuals

Distance violation markers

(You can add screenshots here later.)

🤝 Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests.

📜 License

This project is released under the MIT License.
You may modify and use the source code freely.

If you'd like, I can also:

✅ add badges (build status, license, Python version)
✅ create a project logo
✅ generate screenshots or diagrams
✅ help clean your script for better code quality
