import cv2
import numpy as np
import colorsys

# Video setup
width, height = 1280, 720
fps = 30
frames_per_transition = 30
n = 6  # Number of words
layers = 5
words = ["The", "cat", "sat", "on", "the", "mat"]
theta = 5 * np.pi / 180  # 5-degree rotation

# Rotation matrix for values
W_v = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

# Initial embeddings on a circle
angles = -np.pi / 2 + 2 * np.pi * np.arange(n) / n
X0 = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (n, 2)

# Precompute embeddings across layers
X_history = [X0]
X_current = X0
for _ in range(layers):
    Q = X_current
    K = X_current
    V = X_current @ W_v
    scores = np.exp(Q @ K.T / np.sqrt(2))
    scores /= scores.sum(axis=1, keepdims=True)
    X_next = scores @ V
    X_history.append(X_next)
    X_current = X_next

# Color mapping function (blue to red)
def get_color(weight):
    hue = (2/3) * (1 - weight)  # Blue (2/3) to red (0)
    rgb = colorsys.hsv_to_rgb(hue, 1, 1)
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

# Initialize video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('self_attention_latent_space.mp4', fourcc, fps, (width, height))

# Title screen (2 seconds)
for _ in range(60):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    title = "Self-Attention in Latent Space"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(title, font, 2, 3)[0]
    cv2.putText(img, title, (640 - text_size[0] // 2, 360 + text_size[1] // 2), font, 2, (0, 0, 0), 3)
    video.write(img)

# Animation
for l in range(layers):
    for f in range(frames_per_transition):
        alpha = f / frames_per_transition
        X_f = (1 - alpha) * X_history[l] + alpha * X_history[l + 1]
        
        # Dynamic attention scores
        Q_f = X_f
        K_f = X_f
        scores_f = np.exp(Q_f @ K_f.T / np.sqrt(2))
        scores_f /= scores_f.sum(axis=1, keepdims=True)
        
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw attention lines
        for i in range(n):
            for j in range(n):
                if i != j:
                    weight = scores_f[i, j]
                    color = get_color(weight)
                    thickness = max(1, int(5 * weight))
                    start = (int(640 + X_f[i, 0] * 200), int(360 - X_f[i, 1] * 200))
                    end = (int(640 + X_f[j, 0] * 200), int(360 - X_f[j, 1] * 200))
                    cv2.line(img, start, end, color, thickness)
        
        # Draw points and labels
        for i in range(n):
            pos = (int(640 + X_f[i, 0] * 200), int(360 - X_f[i, 1] * 200))
            cv2.circle(img, pos, 10, (255, 255, 255), -1)  # White fill
            cv2.circle(img, pos, 10, (0, 0, 0), 1)        # Black border
            cv2.putText(img, words[i], (pos[0] + 15, pos[1] + 5), font, 0.7, (0, 0, 0), 2)
        
        # Color bar
        for k in range(100, 1180):
            weight = (k - 100) / (1180 - 100)
            color = get_color(weight)
            cv2.line(img, (k, 650), (k, 680), color, 1)
        cv2.putText(img, "Low attention", (100, 710), font, 0.7, (0, 0, 0), 1)
        text_size = cv2.getTextSize("High attention", font, 0.7, 1)[0]
        cv2.putText(img, "High attention", (1180 - text_size[0], 710), font, 0.7, (0, 0, 0), 1)
        
        video.write(img)

video.release()
print("Video 'self_attention_latent_space.mp4' has been created.")