# ASTRO — Coordinate Frames & Kinematic Transformation Specification

## 1. Coordinate Frame Reference Standards (REP-103)

All robot spatial frames adhere to **ROS REP-103**:
- $+X$: Forward
- $+Y$: Left
- $+Z$: Up
- Yaw ($\theta$): Counter-Clockwise (CCW) rotation around $+Z$ (positive = turn left, negative = turn right).

Optical frames (Camera sensor standard):
- $+X_{opt}$: Right
- $+Y_{opt}$: Down
- $+Z_{opt}$: Forward (Depth axis)

---

## 2. Rigid Sensor Tree & Joint Kinematics

Both the **ReSpeaker v3.0 4-mic array** and the **OAK-D Lite camera** are rigidly attached to `head_link`. As `head_yaw_joint` rotates, both sensor frames rotate with it.

```mermaid
graph TD
    BaseLink["base_link (Robot Base)"] -->|head_yaw_joint (θ)| HeadLink["head_link (Yaw Angle θ)"]
    HeadLink -->|dx=0.06m, dz=0.02m| OakLink["oak_rgb_camera_optical_frame"]
    HeadLink -->|dx=0.00m, dz=0.05m| MicLink["mic_link"]
```

### Physical Sensor Offsets:
- **`head_link` from `base_link`**: $(x=0.00\text{ m}, y=0.00\text{ m}, z=0.21\text{ m})$
- **`oak_link` from `head_link`**: $(x=0.06\text{ m}, y=0.00\text{ m}, z=0.02\text{ m})$
- **`mic_link` from `head_link`**: $(x=0.00\text{ m}, y=0.00\text{ m}, z=0.05\text{ m})$

---

## 3. Mathematical Transformation Formulations

### 3.1. Acoustic DOA Transformation
ReSpeaker firmware outputs raw sound azimuth $\alpha_{raw} \in [0^\circ, 360^\circ)$ clockwise (0°=Front, 90°=Right, 270°=Left).
1. Convert clockwise azimuth to head-relative REP-103 angle:
   $$\theta_{rel} = \text{wrap\_deg}\left(-(\alpha_{raw} + \Delta\alpha_{calib})\right)$$
2. Transform head-relative bearing to robot body frame:
   $$\theta_{body} = \text{wrap\_deg}\left(\theta_{head} + \theta_{rel}\right)$$

### 3.2. Optical 2D Pinhole to 3D Camera Frame
For a 2D bounding box center $(u, v)$ in pixels and metric stereo depth $z_{depth}$:
$$x_{opt} = \frac{(u - c_x) \cdot z_{depth}}{f_x}$$
$$y_{opt} = \frac{(v - c_y) \cdot z_{depth}}{f_y}$$
$$z_{opt} = z_{depth}$$

### 3.3. 3D Camera Frame to Robot Body Base Frame
Transforming optical point $(x_{opt}, y_{opt}, z_{opt})$ to `head_link` coordinates $(x_h, y_h, z_h)$:
$$x_h = z_{opt} + 0.06$$
$$y_h = -x_{opt}$$
$$z_h = -y_{opt} + 0.02$$

Rotating by actual head yaw $\theta_{head}$ around $+Z$:
$$x_{base} = x_h \cos\theta_{head} - y_h \sin\theta_{head}$$
$$y_{base} = x_h \sin\theta_{head} + y_h \cos\theta_{head}$$
$$z_{base} = z_h + 0.21$$

Body-frame azimuth angle:
$$\theta_{body} = \text{atan2}(y_{base}, x_{base})$$
