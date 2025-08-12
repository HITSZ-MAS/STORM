<h1 align="center" style="font-size: 3em;">STORM: Spatial-Temporal Iterative Optimization for Reliable Multicopter Trajectory Generation</h1>
<h3 align="center" style="font-size: 3em;">Accepted by IROS 2025</h3>
<!-- <h2 align="center" style="font-size: 3em;">Code is coming soon...</h2> -->

### Algorithm Framework

<div align="center">
  <img src="files/structure.png" alt="Project Structure" width="100%">
</div>

### Simulation Experiments
<!-- <p align="center">
  <img src="files/simulation1.gif" alt="simulation1" style="width: 48%; height: auto;">
  <img src="files/simulation2.gif" alt="simulation2" style="width: 48%; height: auto;">
</p> -->

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="files/simulation1.gif" alt="simulation1" width="100%">
      </td>
      <td align="center">
        <img src="files/simulation2.gif" alt="simulation2" width="100%">
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="files/simulation3.gif" alt="simulation3" width="100%">
      </td>
      <td align="center">
        <img src="files/simulation4.gif" alt="simulation4" width="100%">
      </td>
    </tr>
  </table>
</div>

### Real-World Experiments
<p align="center">
  <img src="files/real.gif" alt="real world" style="width: 100%; height: auto;">
</p>

Complete video: 
[video](https://www.bilibili.com/video/BV18Y9nYJE2z/)

To learn more about our project, please refer to our paper on arXiv:

- **[STORM: Spatial-Temporal Iterative Optimization for Reliable Multicopter Trajectory Generation](https://arxiv.org/abs/2503.03252)**

## System Architecture

```
BSpline.py              - B-spline base class and coefficient matrix calculation
cps_optimizer.py        - Control points optimizer
knots_optimizer.py      - Knot time optimizer
optimizer.py            - Iterative optimizer main class
opt_dataloader.py       - ROS data loading and visualization node
```

## Dependencies

### Required Packages
```bash
# Core numerical computation
numpy>=1.19.0
scipy>=1.5.0

# Optimization solver
cvxopt>=1.2.0

# Symbolic computation (for linearization optimization)
sympy>=1.7.0

# ROS related (only required for opt_dataloader.py)
rospy>=1.15.0
visualization_msgs
geometry_msgs
std_msgs

# Message types (need to compile ROS package)
planner.msg
```

### Install Dependencies
```bash
pip install numpy scipy cvxopt sympy
# ROS related packages need to be installed via catkin_make or colcon build
```

## Input Format

### 1. Control Points Input (opt_dataloader.py)
```python
# ROS message format
ctrlpts_msg.x = [x1, x2, ..., xn]  # x coordinate list
ctrlpts_msg.y = [y1, y2, ..., yn]  # y coordinate list  
ctrlpts_msg.z = [z1, z2, ..., zn]  # z coordinate list
```

### 2. Corridor Constraints Input
```python
# Each corridor segment contains multiple plane constraints
corridor_msg.corridors = [
    {
        'a': [a1, a2, ...],  # plane normal vector x component
        'b': [b1, b2, ...],  # plane normal vector y component
        'c': [c1, c2, ...],  # plane normal vector z component
        'd': [d1, d2, ...]   # plane distance parameter
    },
    # ... more corridor segments
]
```

### 3. Index Input
```python
idx_msg.start_idx = [idx1, idx2, ...]  # corridor segment start index
```

## Output Format

### 1. Optimized Trajectory
```python
# Returns numpy array with shape (n, 3)
trajectory = [
    [x1, y1, z1],  # 1st trajectory point
    [x2, y2, z2],  # 2nd trajectory point
    ...
    [xn, yn, zn]   # nth trajectory point
]
```

### 2. ROS Message Output
```python
# B-spline state message
bspline_msg = {
    'x': [x1, x2, ...],           # trajectory x coordinates
    'y': [y1, y2, ...],           # trajectory y coordinates
    'z': [z1, z2, ...],           # trajectory z coordinates
    'vx': [vx1, vx2, ...],        # velocity x component
    'vy': [vy1, vy2, ...],        # velocity y component
    'vz': [vz1, vz2, ...],        # velocity z component
    'ax': [ax1, ax2, ...],        # acceleration x component
    'ay': [ay1, ay2, ...],        # acceleration y component
    'az': [az1, az2, ...],        # acceleration z component
    'jx': [jx1, jx2, ...],        # jerk x component
    'jy': [jy1, jy2, ...],        # jerk y component
    'jz': [jz1, jz2, ...],        # jerk z component
    'duration': total_time         # total time
}
```

### 2. Using ROS Node
```bash
# Start ROS node
rosrun planner opt_dataloader.py

# Publish input data
rostopic pub /local_planning/init_ctrlpts planner/ctrlpts "..." 
rostopic pub /local_planning/corridor planner/corridor "..."
rostopic pub /local_planning/idx planner/idx "..."
```




### Get in Touch

**For any inquiries or more information, please feel free to contact us at:**  
**24S053067@stu.hit.edu.cn**

Please kindly star this project and cite out paper if it helps you. Thank you!
```
@article{zhang2025storm,
  title={STORM: Spatial-Temporal Iterative Optimization for Reliable Multicopter Trajectory Generation},
  author={Zhang, Jinhao and Zhou, Zhexuan and Xia, Wenlong and Gong, Youmin and Mei, Jie},
  journal={arXiv preprint arXiv:2503.03252},
  year={2025}
}
```

*Thank you for your interest!*


