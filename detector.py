import cv2
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def check_region(x, y, video):
    r0: int = 0
    r1: int = 0
    r2: int = 0

    width = video.get(3)
    height = video.get(4)

    upper_left0 = (0, 0)
    bottom_right0 = (width / 3 - 50, height)
    upper_left1 = (220, 0)
    bottom_right1 = (width / 3 + 220 - 50, height)
    upper_left2 = (440, 0)
    bottom_right2 = (width / 3 + 440 - 50, height)

    if (upper_left0[0] < x < upper_left0[0] + width) and (upper_left0[1] < y < height): r0 = 1
    if (upper_left1[0] < x < upper_left1[0] + width) and (upper_left1[1] < y < height): r1 = 1
    if (upper_left2[0] < x < upper_left2[0] + width) and (upper_left2[1] < y < height): r2 = 1

    return r0, r1, r2


def update(frame, x, y, ln):
    x.append(x[-1] + 1)
    y.append(1)

    ln.set_data(x, y)
    plt.show()
    return ln


def get_center(rect):
    x, y, w, h = cv2.boundingRect(rect)
    return (x + x + w) / 2, (y + y + h) / 2


def motion_detection():
    fig, (all, ax0, ax1, ax2) = plt.subplots(4, 1, figsize=(9, 9))

    x_axis = [0]
    y_axis = [1]

    r0_y = [1]
    r1_y = [1]
    r2_y = [1]

    r0 = 1
    r1 = 1
    r2 = 1

    initialState = None
    threshold = 900

    video = cv2.VideoCapture("ball.mp4")
    start_time = time.time()  # start timer
    var_motion = 0

    # plt.ion()

    while True:
        ret, cur_frame = video.read()
        if ret:


            # r0_y.append(r0)  # append to graphs
            # r1_y.append(r1)
            # r2_y.append(r2)


            gray_image = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_image, (21, 21), 0)

            if initialState is None:
                initialState = gray_frame
                continue

            differ_frame = cv2.absdiff(initialState, gray_frame)
            thresh_frame = cv2.threshold(differ_frame, 30, 255, cv2.THRESH_BINARY)[1]
            thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

            # Finding contours
            cont, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cur in cont:
                if cv2.contourArea(cur) < threshold:
                    r0 = 0
                    r1 = 0
                    r2 = 0
                    var_motion = 0
                    continue

                var_motion = 1
                (cur_x, cur_y, cur_w, cur_h) = cv2.boundingRect(cur)
                cv2.rectangle(cur_frame, (cur_x, cur_y), (cur_x + cur_w, cur_y + cur_h), (0, 255, 0), 1)


                point_x, point_y = get_center(cur)
                r0, r1, r2 = check_region(point_x, point_y, video)  # check in which part of frame there was motion

            x_axis.append(time.time() - start_time)
            y_axis.append(var_motion)

            r0_y.append(r0)  # append to graphs
            r1_y.append(r1)
            r2_y.append(r2)

            frame0, frame1, frame2 = divide_frame(cur_frame, video.get(3), video.get(4))
            cur_frames = cv2.resize(cur_frame, (1000, 480))
            frame0s = cv2.resize(frame0, (375, 375))
            frame1s = cv2.resize(frame1, (375, 375))
            frame2s = cv2.resize(frame2, (375, 375))


            cv2.imshow("frame_all", cur_frames)
            cv2.moveWindow("frame_all", 0, 0)

            cv2.imshow("left", frame0s)
            cv2.moveWindow("left", 0, 600)

            cv2.imshow("center", frame1s)
            cv2.moveWindow("center", 700, 600)

            cv2.imshow("right", frame2s)
            cv2.moveWindow("right", 1400, 600)
            cv2.waitKey(1)

            # animation = FuncAnimation(fig, update(fig, x_axis, y_axis, ln), interval=500)

            continue
        else:
            break
    # plt.ioff()  # due to infinite loop, this gets never called.
    # plt.show()
    video.release()
    cv2.destroyAllWindows()

    all.plot(x_axis, y_axis, '-', color='black')
    all.set_ylabel('frame', size=15)
    ax0.plot(x_axis, r0_y, '-', color='red')
    ax0.set_ylabel('left', size=15)
    ax1.plot(x_axis, r1_y, '-', color='green')
    ax1.set_ylabel('center', size=15)
    ax2.plot(x_axis, r2_y, '-', color='blue')
    ax2.set_ylabel('right', size=15)


    plt.show()


def divide_frame(frame, width, height):
    width = int(width)
    height = int(height)

    upper_left0 = (0, 0)
    bottom_right0 = (int(width / 3) - 50, height)

    upper_left1 = (220, 0)
    bottom_right1 = (int(width / 3) + 220 - 50, height)

    upper_left2 = (440, 0)
    bottom_right2 = (int(width / 3) + 440 - 50, height)

    frame0 = frame[upper_left0[1]: bottom_right0[1], upper_left0[0]: bottom_right0[0]]
    frame1 = frame[upper_left1[1]: bottom_right1[1], upper_left1[0]: bottom_right1[0]]
    frame2 = frame[upper_left2[1]: bottom_right2[1], upper_left2[0]: bottom_right2[0]]

    # cv2.imshow("frame0", frame0)
    # cv2.imshow("frame1", frame1)
    # cv2.imshow("frame2", frame2)

    return frame0, frame1, frame2


def sketch_transform(image):
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (7, 7), 0)
    image_canny = cv2.Canny(image_grayscale_blurred, 10, 80)
    _, mask = image_canny_inverted = cv2.threshold(image_canny, 30, 255, cv2.THRESH_BINARY_INV)

    return mask


def this_right_now():
    cam_capture = cv2.VideoCapture("arad.mp4")
    cv2.destroyAllWindows()

    width = cam_capture.get(3)  # float `width`
    height = cam_capture.get(4)  # float `height`

    width = int(width)
    height = int(height)
    upper_left0 = (0, 0)
    bottom_right0 = (int(width / 3) - 50, height)
    upper_left1 = (220, 0)
    bottom_right1 = (int(width / 3) + 220 - 50, height)
    upper_left2 = (440, 0)
    bottom_right2 = (int(width / 3) + 440 - 50, height)

    while True:
        _, image_frame = cam_capture.read()

        frame0 = image_frame
        frame1 = image_frame
        frame2 = image_frame

        # PART 1: divide window into 3 parts.

        # Rectangle marker 1
        r0 = cv2.rectangle(image_frame, upper_left0, bottom_right0, (100, 50, 200), 5)
        rect_img0 = image_frame[upper_left0[1]: bottom_right0[1], upper_left0[0]: bottom_right0[0]]
        sketcher_rect0 = rect_img0
        sketcher_rect0 = sketch_transform(sketcher_rect0)
        # sketcher_rect0 = motion_detection(sketcher_rect0)
        sketcher_rect_rgb0 = cv2.cvtColor(sketcher_rect0, cv2.COLOR_GRAY2RGB)
        image_frame[upper_left0[1]: bottom_right0[1], upper_left0[0]: bottom_right0[0]] = sketcher_rect_rgb0
        frame0 = frame0[upper_left0[1]: bottom_right0[1], upper_left0[0]: bottom_right0[0]]

        # Rectangle marker 2
        r1 = cv2.rectangle(image_frame, upper_left1, bottom_right1, (100, 50, 200), 5)
        rect_img1 = image_frame[upper_left1[1]: bottom_right1[1], upper_left1[0]: bottom_right1[0]]
        sketcher_rect1 = rect_img1
        sketcher_rect1 = sketch_transform(sketcher_rect1)
        sketcher_rect_rgb1 = cv2.cvtColor(sketcher_rect1, cv2.COLOR_GRAY2RGB)
        frame1 = frame1[upper_left0[1]: bottom_right0[1], upper_left0[0]: bottom_right0[0]]

        # Rectangle marker 3
        r2 = cv2.rectangle(image_frame, upper_left2, bottom_right2, (100, 50, 200), 5)
        rect_img2 = image_frame[upper_left2[1]: bottom_right2[1], upper_left2[0]: bottom_right2[0]]
        sketcher_rect2 = rect_img2
        sketcher_rect2 = sketch_transform(sketcher_rect2)
        sketcher_rect_rgb2 = cv2.cvtColor(sketcher_rect2, cv2.COLOR_GRAY2RGB)
        frame2 = frame2[upper_left0[1]: bottom_right0[1], upper_left0[0]: bottom_right0[0]]

        cv2.imshow("Sketcher ROI", image_frame)
        cv2.imshow("frame0", frame0)
        cv2.imshow("frame1", frame1)
        cv2.imshow("frame2", frame2)

        if cv2.waitKey(1) == 13:
            break

        # PART 2: insert motion detection.

    cam_capture.release()
    cv2.destroyAllWindows()


def main():
    motion_detection()
    # draw_graph()


if __name__ == "__main__":
    main()
