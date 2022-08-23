#!/usr/bin/python3
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'tesseract'
print(pytesseract.get_languages(config=''))

HEADLINE_AVG_COLOR = (129.5148472, 62.9367192, 53.23520085)  # BGR
os.system('ffmpeg -y -i ' + sys.argv[1] + '-an -vf "crop=1448:130:327:832" 2.' + sys.argv[1])
os.system('ffmpeg -y -i 2.' + sys.argv[1] + '-vf mpdecimate,setpts=N/FRAME_RATE/TB ' + sys.argv[1])

def extract_headline(frame, do_ocr):
    """Extract headline from a single frame"""
    #h, w, _ = frame.shape
    #x1 = int(w * 0.17)
    #x2 = int(w * 0.925)
    #y1 = int(h * 0.77)
    #y2 = int(h * 0.89)

    headline = ''
    headline_like_frame_detected = False
    headline_img = frame[0:1448, 0:832]

    # Get avg color and compare it to a headline avg color
    current_avg_color = np.average(np.average(headline_img, axis=0), axis=0)
    col_diff = abs(np.subtract(HEADLINE_AVG_COLOR, current_avg_color))

    # if bgr channels are less different than 5 (in scale 0-255) then it's probably a headline
    if col_diff[0] < 10 and col_diff[1] < 10 and col_diff[2] < 10:
        print('Color diff:', col_diff)
        headline_like_frame_detected = True
        if do_ocr:
            headline_gray = cv2.cvtColor(headline_img, cv2.COLOR_BGR2GRAY)
            headline = pytesseract.image_to_string(headline_gray, config='--psm 6', lang='pol')

    return headline_img, headline, headline_like_frame_detected


def tvp_headlines_mp4(input_video_path, output_headlines_path):
    """Extract all headlines from a video file"""
    cap = cv2.VideoCapture(input_video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(width, height, fps)

    headlines = []
    frame_count = 0
    last_headline_frame = 0
    headline_animation_start_frame = 0
    has_headline_animation_count_started = False
    do_ocr = False
    SECONDS_TO_SKIP_AFTER_LAST_HEADLINE = 20
    HEADLINE_ANIMATION_SECONDS = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip ocr if last headline was seen 20s ago
        if frames_to_seconds(fps, frame_count - last_headline_frame) > SECONDS_TO_SKIP_AFTER_LAST_HEADLINE:

            # Wait for headline animation to end (2S)
            if frames_to_seconds(fps, frame_count - headline_animation_start_frame) > HEADLINE_ANIMATION_SECONDS and has_headline_animation_count_started:
                do_ocr = True

            # Extract headline with ocr
            headline_img, headline, headline_like_frame_detected = extract_headline(frame, do_ocr)

            # If extract_headline stopped detecting headline-like frames then stop animation counters (headline-like frames must be detected every frame for at least 2 seconds)
            if has_headline_animation_count_started and not headline_like_frame_detected:
                has_headline_animation_count_started = False

            # If headline-like frame was detected and animation counter hasn't started yet then start it
            if headline_like_frame_detected and not has_headline_animation_count_started:
                has_headline_animation_count_started = True
                headline_animation_start_frame = frame_count

            # Save headline, reset counters
            if headline != '':
                print('Frame:', frame_count, 'Headline:', headline)
                last_headline_frame = frame_count
                headlines.append(headline)
                do_ocr = False
                has_headline_animation_count_started = False
                headline_animation_start_frame = 0

                save_headline(headline, output_headlines_path)

        time_elapsed = frames_to_seconds(fps, frame_count)
        last_headline_elapsed = frames_to_seconds(fps, frame_count - last_headline_frame)
        animation_time_elapsed = 0 if headline_animation_start_frame == 0 else frames_to_seconds(fps, frame_count - headline_animation_start_frame)

        frame_count += 1

        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(headlines)


def frames_to_seconds(fps, frame_count):
    return frame_count / fps


def save_headline(headline, path):
    with open(path, 'a') as f:
        f.write(headline.strip('\n') + '\n')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_video_path = sys.argv[1]
    output_headlines_path = sys.argv[2]

    tvp_headlines_mp4(input_video_path, output_headlines_path)
