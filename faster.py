from matplotlib import pyplot as plt
import mediapipe as mp
import cv2
from mediapipe.framework.formats import landmark_pb2
import math
from ControlBilibili import ControlBilibili

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})

mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles


def display_one_image(image, title, subplot, titlesize=16):
    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)


def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
    # Images and labels.
    images = [image.numpy_view() for image in images]
    gestures = [top_gesture for (top_gesture, _) in results]
    multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    # Display gestures and hand landmarks.
    for i, (image, gestures) in enumerate(zip(images[:rows*cols], gestures[:rows*cols])):
        title = f"{gestures.category_name} ({gestures.score:.2f})"
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3
        annotated_image = image.copy()

        for hand_landmarks in multi_hand_landmarks_list[i]:
          hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
          ])

          # mp_drawing.draw_landmarks(
          #   annotated_image,
          #   hand_landmarks_proto,
          #   mp_hands.HAND_CONNECTIONS,
          #   mp_drawing_styles.get_default_hand_landmarks_style(),
          #   mp_drawing_styles.get_default_hand_connections_style())

        subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

    # Layout.
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()

def main():
    base_options = mp.tasks.BaseOptions(model_asset_path='gesture_recognizer.task')
    options = mp.tasks.vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

    images = []
    results = []
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        else:
            # STEP 3: Load the input image.
            #image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            # STEP 4: Recognize gestures in the input image.
            recognition_result = recognizer.recognize(image)

            # STEP 5: Process the result. In this case, visualize it.
            images.append(image)
            if recognition_result.gestures:
                top_gesture = recognition_result.gestures[0][0]
            else:
                continue

            #top_gesture = recognition_result.gestures
            print(top_gesture)
            gesture_name=top_gesture.category_name
            print(gesture_name)
            # list=[]
            # list=list(top_gesture)
            # print(type(list),"\n",list)
            # hand_landmarks = recognition_result.hand_landmarks
            # results.append((top_gesture, hand_landmarks))

            #display_batch_of_images_with_gestures_and_hand_landmarks(images, results)
            if gesture_name == "Thumb_Up":#点赞
                ControlBilibili.praise()
                print(top_gesture)
            elif gesture_name == "Open_Palm":
                ControlBilibili.stopAndPlayVideo()
                print(top_gesture)
            elif gesture_name == "Thumb_Down":
                ControlBilibili.speedDown()
            elif gesture_name == "Closed_Fist":
                ControlBilibili.mute()
            elif gesture_name=="Pointing_Up":
                ControlBilibili.speedUp()
            elif gesture_name=="Victory":
                ControlBilibili.nextVideo()
            elif gesture_name=="ILoveYou":
                ControlBilibili.FullScreen()
            else:
                continue
main()


