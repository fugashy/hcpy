import cv2
import os
import requests
import click

FACE_CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
BODY_CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_fullbody.xml'
PLATE_CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml'

EXT_IMAGES = ['.jpg', '.jpeg', '.png', '.bmp']
EXT_VIDEOS = ['.mp4', '.avi', '.mov', '.mkv']

COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        ]


def download_cascade_file(url, filename):
    if os.path.exists(filename):
        return

    print(f"Downloading {filename}...")
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)
    print(f"{filename} downloaded.")


def _detect(image, classifiers):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rectangles = list()
    for classifier in classifiers:
        rectangles.append(
                classifier.detectMultiScale(
                    gray_image,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(10, 10)))
    return rectangles


def process_image(input_file, classifiers):
    image = cv2.imread(input_file)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return list()

    rectangles_list = _detect(image, classifiers)

    for rectangles, color in zip(rectangles_list, COLORS):
        for (x, y, w, h) in rectangles:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 5)

    # 結果を表示
    cv2.imshow('Detected Faces and Plates (Image)', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(input_file, classifiers):
    video_capture = cv2.VideoCapture(input_file)

    if not video_capture.isOpened():
        print(f"Error: Could not open video {input_file}")
        return

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break
        rectangles_list = _detect(frame, classifiers)

        for rectangles, color in zip(rectangles_list, COLORS):
            for (x, y, w, h) in rectangles:
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 5)

        cv2.imshow('Detected Faces and Plates', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()



@click.command()
@click.argument('input_file', type=click.Path(exists=False))
@click.option("--face", is_flag=True, default=False)
@click.option("--number-plate", is_flag=True, default=False)
@click.option("--body", is_flag=True, default=False)
def entry_point(input_file, face, number_plate, body):
    classifiers = list()
    if face:
        cascade_path = '/tmp/haarcascade_frontalface_default.xml'
        download_cascade_file(FACE_CASCADE_URL, cascade_path)
        classifiers.append(
                cv2.CascadeClassifier(cascade_path))
    if number_plate:
        cascade_path = '/tmp/haarcascade_russian_plate_number.xml'
        download_cascade_file(PLATE_CASCADE_URL, cascade_path)
        classifiers.append(
                cv2.CascadeClassifier(cascade_path))
    if body:
        cascade_path = '/tmp/haarcascade_fullbody.xml'
        download_cascade_file(BODY_CASCADE_URL, cascade_path)
        classifiers.append(
                cv2.CascadeClassifier(cascade_path))

    if len(classifiers) == 0:
        print("no classifiers")
        return

    ext = os.path.splitext(input_file)[1].lower()
    process_func = None
    if ext in EXT_IMAGES:
        process_func = process_image
    elif ext in EXT_VIDEOS:
        process_func = process_video
    elif len(input_file) == 1:
        input_file = int(input_file)
        process_func = process_video


    if process_func is None:
        print(f"no supported ext: {ext}")
        return

    process_func(input_file, classifiers)
