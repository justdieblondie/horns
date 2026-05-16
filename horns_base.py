import cv2
import mediapipe as mp
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
import time

# --- Zmienne globalne ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh_tessellation = mp_face_mesh.FACEMESH_TESSELATION

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

points_3d = np.array([])
horn_points_current = np.array([])
horn_connections = []
smoothed_landmarks = np.array([])

SMOOTHING_ALPHA = 0.2
horn_state = 'HIDDEN'
animation_progress = 0.0
ANIMATION_SPEED = 0.03

MODEL_SCALE = 3.0
VIEW_SCALE = 4.0
cap = None

frame_count = 0
start_time = time.time()
actual_fps = 0

x_rot = 0.0
y_rot = 0.0
last_mouse_x = 0
last_mouse_y = 0
is_mouse_down = False
MOUSE_SENSITIVITY = 0.2

video_texture_id = None

def init_video_texture(width, height):
    global video_texture_id
    video_texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, video_texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # Alokujemy teksturę, która będzie przechowywać dane w formacie RGB
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glBindTexture(GL_TEXTURE_2D, 0)

def setup_opengl(width, height):
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POINT_SMOOTH)
    glPointSize(1.0)
    glLineWidth(1.0)

def get_bezier_point(p0, p1, p2, t): return (1-t)**2*p0 + 2*(1-t)*t*p1 + t**2*p2
def get_bezier_tangent(p0, p1, p2, t):
    vec = 2*(1-t)*(p1-p0) + 2*t*(p2-p1)
    norm = np.linalg.norm(vec); return vec/norm if norm > 0 else np.array([0,1,0])

def generate_thick_devil_horns(landmarks_3d):
    required_indices = [10, 103, 168, 234, 332, 454]
    if any(i >= len(landmarks_3d) for i in required_indices): return [], []
    p_up_top, p_up_bottom = landmarks_3d[10], landmarks_3d[168]
    head_up = (p_up_top - p_up_bottom) / np.linalg.norm(p_up_top - p_up_bottom)
    p_right_side, p_left_side = landmarks_3d[454], landmarks_3d[234]
    head_right = (p_right_side - p_left_side) / np.linalg.norm(p_right_side - p_left_side)
    structure, connections = [], []
    HORN_SCALE, MAX_RADIUS, SPINE_STEPS, CIRCLE_POINTS = MODEL_SCALE*0.3, MODEL_SCALE*0.04, 22, 7
    for side_idx, side in enumerate([-1, 1]):
        base_idx = 103 if side == -1 else 332
        p_base = landmarks_3d[base_idx]
        p0, p1, p2 = p_base, \
            p_base + (head_up*1. - np.cross(head_right, head_up)*1.2 + head_right*side*0.3)*HORN_SCALE, \
            p_base + (head_up*2. - np.cross(head_right, head_up)*0.8 + head_right*side*0.1)*HORN_SCALE
        offset = side_idx * (SPINE_STEPS + 1) * CIRCLE_POINTS
        for i in range(SPINE_STEPS + 1):
            t = i/float(SPINE_STEPS)
            spine_point = get_bezier_point(p0, p1, p2, t)
            tangent_vec = get_bezier_tangent(p0, p1, p2, t)
            perp_vec1 = np.cross(tangent_vec, head_right); perp_vec1 /= np.linalg.norm(perp_vec1)
            perp_vec2 = np.cross(tangent_vec, perp_vec1); perp_vec2 /= np.linalg.norm(perp_vec2)
            radius = MAX_RADIUS * math.pow(1.0 - t, 0.7)
            for j in range(CIRCLE_POINTS):
                theta = (j/float(CIRCLE_POINTS))*2*math.pi
                final_point = spine_point + radius*(perp_vec1*math.cos(theta) + perp_vec2*math.sin(theta))
                structure.append((p_base, final_point))
                current_pt_idx = i*CIRCLE_POINTS+j+offset
                if j < CIRCLE_POINTS-1: connections.append((current_pt_idx, current_pt_idx + 1))
                else: connections.append((current_pt_idx, i*CIRCLE_POINTS+offset))
                if i < SPINE_STEPS: connections.append((current_pt_idx, current_pt_idx + CIRCLE_POINTS))
    return structure, connections

def display():
    global actual_fps
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    window_w = glutGet(GLUT_WINDOW_WIDTH)
    window_h = glutGet(GLUT_WINDOW_HEIGHT)

    # === CZĘŚĆ 1: Rysowanie modelu 3D w górnej połowie okna ===
    glViewport(0, window_h // 2, window_w, window_h // 2)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect_ratio = (window_h / 2) / window_w if window_w > 0 else 1.0
    view_height = VIEW_SCALE * aspect_ratio
    glOrtho(-VIEW_SCALE/2., VIEW_SCALE/2., -view_height/2., view_height/2., -100.0, 100.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0.0, -0.5, 0.0)
    glRotatef(x_rot, 1.0, 0.0, 0.0)
    glRotatef(y_rot, 0.0, 1.0, 0.0)

    glColor3f(1.0, 0.0, 0.0)
    if points_3d.any():
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, points_3d)
        glDrawArrays(GL_POINTS, 0, len(points_3d))
        glDisableClientState(GL_VERTEX_ARRAY)

    if horn_points_current.any():
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, horn_points_current)
        glDrawArrays(GL_POINTS, 0, len(horn_points_current))
        glDisableClientState(GL_VERTEX_ARRAY)

    # === CZĘŚĆ 2: Rysowanie klatki z kamery w dolnej połowie okna (Z KOREKTĄ PROPORCJI) ===
    glViewport(0, 0, window_w, window_h // 2)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, window_w, 0, window_h // 2)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    viewport_w = window_w
    viewport_h = window_h // 2

    video_ar = video_w / video_h if video_h > 0 else 1.0
    viewport_ar = viewport_w / viewport_h if viewport_h > 0 else 1.0

    if viewport_ar > video_ar:
        new_h = viewport_h
        new_w = int(new_h * video_ar)
        x_offset = (viewport_w - new_w) // 2
        y_offset = 0
    else:
        new_w = viewport_w
        new_h = int(new_w / video_ar)
        x_offset = 0
        y_offset = (viewport_h - new_h) // 2

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, video_texture_id)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(x_offset, y_offset)
    glTexCoord2f(1, 1); glVertex2f(x_offset + new_w, y_offset)
    glTexCoord2f(1, 0); glVertex2f(x_offset + new_w, y_offset + new_h)
    glTexCoord2f(0, 0); glVertex2f(x_offset, y_offset + new_h)
    glEnd()

    glDisable(GL_TEXTURE_2D)

    glWindowPos2i(10, 10)
    glColor3f(0.0, 1.0, 0.0)
    for char in f"FPS: {actual_fps:.2f}":
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))

    glutSwapBuffers()

def keyboard(key, x, y):
    global horn_state, VIEW_SCALE

    ZOOM_SPEED = 0.2
    if key == b'b':
        horn_state = 'GROWING' if horn_state in ['HIDDEN','SHRINKING'] else 'SHRINKING'
    elif key == b'=' or key == b'+':
        VIEW_SCALE = max(0.2, VIEW_SCALE - ZOOM_SPEED)
    elif key == b'-':
        VIEW_SCALE += ZOOM_SPEED
    elif key == b'\x1b':
        cleanup()

def mouse_button_callback(button, state, x, y):
    global is_mouse_down, last_mouse_x, last_mouse_y
    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            is_mouse_down = True
            last_mouse_x = x
            last_mouse_y = y
        elif state == GLUT_UP:
            is_mouse_down = False

def mouse_motion_callback(x, y):
    global x_rot, y_rot, last_mouse_x, last_mouse_y
    if is_mouse_down:
        dx = x - last_mouse_x
        dy = y - last_mouse_y
        y_rot += dx * MOUSE_SENSITIVITY
        x_rot += dy * MOUSE_SENSITIVITY
        last_mouse_x = x
        last_mouse_y = y

def main_loop():
    global points_3d, horn_points_current, horn_state, animation_progress, smoothed_landmarks, horn_connections, frame_count, start_time, actual_fps

    success, frame = cap.read() # 'frame' jest w formacie BGR
    if not success: return

    h, w, _ = frame.shape
    
    # Do przetwarzania przez MediaPipe wciąż potrzebujemy formatu RGB
    rgb_frame_for_processing = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame_for_processing)

    model_view_height = 0
    if results.multi_face_landmarks:
        current_landmarks_raw = results.multi_face_landmarks[0].landmark
        current_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in current_landmarks_raw])
        
        if smoothed_landmarks.size == 0:
            smoothed_landmarks = current_landmarks
        else:
            smoothed_landmarks = SMOOTHING_ALPHA * current_landmarks + (1.0 - SMOOTHING_ALPHA) * smoothed_landmarks
        
        window_w = glutGet(GLUT_WINDOW_WIDTH)
        window_h = glutGet(GLUT_WINDOW_HEIGHT)
        aspect_ratio = (window_h / 2) / window_w if window_w > 0 else 1.0
        model_view_height = MODEL_SCALE * aspect_ratio
        
        points_3d = np.zeros_like(smoothed_landmarks)
        points_3d[:, 0] = (smoothed_landmarks[:, 0] - 0.5) * MODEL_SCALE
        points_3d[:, 1] = -(smoothed_landmarks[:, 1] - 0.5) * model_view_height
        points_3d[:, 2] = smoothed_landmarks[:, 2] * MODEL_SCALE

    if horn_state == 'GROWING': animation_progress = min(1.0, animation_progress + ANIMATION_SPEED)
    elif horn_state == 'SHRINKING': animation_progress = max(0.0, animation_progress - ANIMATION_SPEED)
    
    if horn_state != 'HIDDEN' and points_3d.any():
        full_horn_structure, local_horn_connections = generate_thick_devil_horns(points_3d)
        if local_horn_connections: horn_connections = local_horn_connections
        
        if full_horn_structure:
            bases, finals = zip(*full_horn_structure)
            bases, finals = np.array(bases), np.array(finals)
            horn_points_current = bases + (finals - bases) * animation_progress
    else:
        horn_points_current = np.array([])

    # # Rysujemy punkty bezpośrednio na klatce 'frame' (która jest w formacie BGR)
    # if smoothed_landmarks.any():
    #     for lm in smoothed_landmarks:
    #         px, py = int(lm[0] * w), int(lm[1] * h)
    #         if 0 < px < w and 0 < py < h:
    #              cv2.circle(frame, (px, py), 1, (0, 255, 0), -1) # landmarks color

    # if horn_points_current.any() and model_view_height > 0:
    #     for p in horn_points_current:
    #         norm_x = (p[0] / MODEL_SCALE) + 0.5
    #         norm_y = -(p[1] / model_view_height) + 0.5
    #         px, py = int(norm_x * w), int(norm_y * h)
    #         if 0 < px < w and 0 < py < h:
    #             cv2.circle(frame, (px, py), 1, (0, 255, 0), -1) #horns color

    # ===  OSTATECZNA POPRAWKA  ===
    # Przesyłamy do tekstury oryginalną klatkę 'frame' (bez dorysowanych punktów)
    # i informujemy OpenGL, że dostarczone dane są w formacie GL_BGR.
    glBindTexture(GL_TEXTURE_2D, video_texture_id)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, frame)
    # === KONIEC POPRAWKI ===
    
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:
        actual_fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    
    glutPostRedisplay()

def cleanup():
    global cap, video_texture_id
    print("Zamykanie...")
    if cap: cap.release()
    if glutGetWindow():
        if video_texture_id:
            glDeleteTextures(1, [video_texture_id])
        glutDestroyWindow(glutGetWindow())

def timer_func(value):
    try:
        main_loop()
        glutTimerFunc(1, timer_func, 0)
    except Exception as e:
        print(f"Błąd w pętli: {e}"); cleanup()

def main():
    global cap, start_time
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kurwa, nie mogę otworzyć kamerki.")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    DESIRED_WIDTH = 800
    DESIRED_HEIGHT = 600
    TARGET_FPS = 60
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Kamera działa w: {w}x{h} @ {fps:.2f} FPS")

    start_time = time.time()

    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(w, h * 2)
    glutCreateWindow(b"OpenGL Horns")

    init_video_texture(w, h)
    setup_opengl(w, h)

    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse_button_callback)
    glutMotionFunc(mouse_motion_callback)
    glutTimerFunc(1, timer_func, 0)
    
    glutCloseFunc(cleanup)
    
    print("Klawisz 'b' -> rogi.")
    print("Klawisze '-' i '=' -> oddalanie/przyblizanie widoku 3D.")
    print("Wciśnij i przytrzymaj lewy przycisk myszy, aby obracać.")
    print("ESC -> zamknij.")
    glutMainLoop()

if __name__ == "__main__":
    main()
