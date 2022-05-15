import cv2
from pathlib import Path

video_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
new_size = 960, 1280

for folder in video_dir.iterdir():
    if folder.is_dir():
        for file in folder.iterdir():
            if file.suffix == '.mp4':
                cap = cv2.VideoCapture(str(file))
                fps = cap.get(cv2.CAP_PROP_FPS)
                out_name = folder.name + '_resized.mp4'
                out = cv2.VideoWriter(
                    str(folder / out_name), fourcc, fps, (new_size))
                while True:
                    ret, frame = cap.read()
                    if ret == True:
                        b = cv2.resize(frame, (new_size), fx=0, fy=0,
                                       interpolation=cv2.INTER_CUBIC)
                        out.write(b)
                    else:
                        break
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                break
