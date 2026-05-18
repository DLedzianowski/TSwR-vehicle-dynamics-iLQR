import numpy as np

def arc(cx, cy, R, start_angle, end_angle, n):
    theta = np.linspace(start_angle, end_angle, n)
    return cx + R * np.cos(theta), cy + R * np.sin(theta)

def line(x_start, x_end, y_start, y_end, n):
    return np.linspace(x_start, x_end, n), np.linspace(y_start, y_end, n)

def generate_reference_path_oval(R=150, L=100, n=100):
    # 1. dolna prosta (lewo -> prawo)
    x1, y1 = line(0, L, -R, -R, n)

    # 2. prawy łuk (dół -> góra, skręt w lewo)
    x2, y2 = arc(L, 0, R, -np.pi/2, np.pi/2, n)

    # 3. górna prosta (prawo -> lewo)
    x3, y3 = line(L, 0, R, R, n)

    # 4. lewy łuk (góra -> dół, skręt w lewo)
    x4, y4 = arc(0, 0, R, np.pi/2, 3*np.pi/2, n)

    x_ref = np.concatenate([x1, x2[1:], x3[1:], x4[1:]])
    y_ref = np.concatenate([y1, y2[1:], y3[1:], y4[1:]])

    return x_ref, y_ref+R

def generate_reference_path_jajo(s):
    x_ref = 75 * np.cos(s) + 75
    y_ref = 50 * np.sin(s)

    return (x_ref, y_ref)

def generate_reference_path_track(s, shift=590): #70
    data = np.loadtxt("traces/Spa_track.csv", delimiter=",", skiprows=1) #1402
    # data = np.loadtxt("traces/Monza_track.csv", delimiter=",", skiprows=1) # 1159
    x_center = data[:, 0]
    y_center = data[:, 1]

    x_center = np.roll(x_center, -shift)
    y_center = np.roll(y_center, -shift)
    # print(f"Reference path length: {len(data)}")
    return x_center, y_center
