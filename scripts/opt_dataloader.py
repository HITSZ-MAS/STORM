#!/usr/bin/env python3
import rospy
import numpy as np
from planner.msg import corridor, ctrlpts, idx, b_spline
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Quaternion
from optimizer import Iter_Optimizer
from std_msgs.msg import ColorRGBA

ctrlpts_input = []
corridors_input = []
idx_input = []
rcv_flag_ctrlpts = False
rcv_flag_corridor = False
rcv_flag_idx = False

ctrlpts_msg = None
corridor_msg = None
idx_msg = None  

######################################### Parameters #########################################
num_sample = 1000
p = 3
V_max = 5
A_max = 3
J_max = 2
decay_factor = 0.3
######################################### Parameters #########################################

vis_pub = rospy.Publisher('local_planning/opt_trajectory', 
                            Marker, 
                            queue_size=1)
bspline_pub = rospy.Publisher('local_planning/bspline_state', b_spline, queue_size=1)
velocity_marker_pub = rospy.Publisher('local_planning/velocity_color_marker', Marker, queue_size=10)


def ctrlpts_callback(msg):
    global ctrlpts_msg
    global rcv_flag_ctrlpts
    rcv_flag_ctrlpts = True
    ctrlpts_msg = msg
def corridor_callback(msg):
    global corridor_msg
    global rcv_flag_corridor
    rcv_flag_corridor = True
    corridor_msg = msg
def idx_callback(msg):
    global idx_msg
    global rcv_flag_idx
    rcv_flag_idx = True
    idx_msg = msg

def process_data(cps, corridors, idx):
    '''
    cps: N*3
    corridors: [M * [P * [4] ] ]
    idx: [M-1]
    '''
    corridor = []
    constraints = []
    for planes in corridors:
        n_vecs = np.array([plane[:3] for plane in planes])
        d_vecs = np.array([plane[3] for plane in planes])
        corridor.append([n_vecs, d_vecs])
    idx = np.array(idx)
    idx[-1] = idx[-1] + 1
    for i, _ in enumerate(cps):
        corridor_idx = np.searchsorted(idx, i, side='right') - 1
        if i in idx[1:-1]:
            constraints.append([[corridor[corridor_idx-1][0], corridor[corridor_idx-1][1]], [corridor[corridor_idx][0], corridor[corridor_idx][1]]])
        else:
            constraints.append([[corridor[corridor_idx][0], corridor[corridor_idx][1]]])

    start_state = cps[0]
    end_state = cps[-1]

    init_guess = cps

    return corridor, constraints, start_state, end_state, init_guess


def visualize_bspline_trajectory(points):
    """
    Visualize B-spline trajectory
    Args:
        trajectory: Optimizer object
    """
    global vis_pub

    traj_vis = Marker()
    traj_vis.header.stamp = rospy.Time.now()
    traj_vis.header.frame_id = "map"
    
    traj_vis.ns = "local_planning/opt_trajectory"
    traj_vis.id = 0
    traj_vis.type = Marker.SPHERE_LIST
    
    traj_vis.action = Marker.ADD
    traj_vis.scale.x = 0.05  # _vis_traj_width
    traj_vis.scale.y = 0.05
    traj_vis.scale.z = 0.05
    
    traj_vis.pose.orientation.x = 0.0
    traj_vis.pose.orientation.y = 0.0
    traj_vis.pose.orientation.z = 0.0
    traj_vis.pose.orientation.w = 1.0
    
    traj_vis.color.a = 1.0
    traj_vis.color.r = 0.0
    traj_vis.color.g = 0.0
    traj_vis.color.b = 1.0
    
    for point in points:
        pt = Point()
        pt.x = point[0]
        pt.y = point[1]
        pt.z = point[2]
        traj_vis.points.append(pt)
    
    vis_pub.publish(traj_vis)

def process_input_data(ctrlpts_msg, corridor_msg, idx_msg):
    ctrlpts_input = []
    for i in range(len(ctrlpts_msg.x)):
        point = [ctrlpts_msg.x[i], ctrlpts_msg.y[i], ctrlpts_msg.z[i]] 
        ctrlpts_input.append(point)
    ctrlpts_input = np.array(ctrlpts_input)
    corridors_input = []
    for corridor_segment in corridor_msg.corridors:
        planes = []
        # Each corridor segment contains multiple planes, each plane is defined by a,b,c,d four parameters
        for i in range(len(corridor_segment.a)):
            plane = [
                corridor_segment.a[i],
                corridor_segment.b[i],
                corridor_segment.c[i],
                corridor_segment.d[i]
            ]
            planes.append(plane)
        corridors_input.append(planes)
    idx_input=idx_msg.start_idx
    return ctrlpts_input, corridors_input, idx_input

def publish_bspline_state(trajectory, velocity, acceleration, jerk, knots):
    """Publish B-spline trajectory, velocity and acceleration, add velocity color visualization"""
    bspline_msg = b_spline()
    bspline_msg.header.stamp = rospy.Time.now()
    bspline_msg.header.frame_id = "map"
    bspline_msg.duration = knots  
    
    # Fill trajectory points
    for i in range(len(trajectory)):
        bspline_msg.x.append(trajectory[i][0])
        bspline_msg.y.append(trajectory[i][1])
        bspline_msg.z.append(trajectory[i][2])
        
        bspline_msg.vx.append(velocity[i][0])
        bspline_msg.vy.append(velocity[i][1])
        bspline_msg.vz.append(velocity[i][2])
        
        bspline_msg.ax.append(acceleration[i][0])
        bspline_msg.ay.append(acceleration[i][1])
        bspline_msg.az.append(acceleration[i][2])
        
        bspline_msg.jx.append(jerk[i][0])
        bspline_msg.jy.append(jerk[i][1])
        bspline_msg.jz.append(jerk[i][2])
    
    bspline_pub.publish(bspline_msg)

    marker = Marker()
    marker.header = bspline_msg.header
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.header.frame_id = "map"
    marker.lifetime = rospy.Duration(0)
    marker.scale.x = 0.1  
    marker.scale.y = 0.1
    marker.pose.orientation = Quaternion(0, 0, 0, 1)
    
    marker.id = 0
    
    max_speed = 5  
    for i in range(len(trajectory)):
        speed = (velocity[i][0]**2 + velocity[i][1]**2 + velocity[i][2]**2)**0.5
        ratio = min(speed / max_speed, 1.0)
        print(f"speed: {speed}, ratio: {ratio}")
        color = ColorRGBA()
        if ratio < 0.5:
            color.r = ratio * 2.0
            color.g = 1.0
            color.b = 0.0
        else:
            color.r = 1.0
            color.g = (1.0 - ratio) * 2.0
            color.b = 0.0
        color.a = 1.0
        
        point = Point()
        point.x = trajectory[i][0]
        point.y = trajectory[i][1]
        point.z = trajectory[i][2]
        marker.points.append(point)
        marker.colors.append(color)  
    
    velocity_marker_pub.publish(marker)

def main():
    rospy.init_node('opt_dataloader', anonymous=True)
    rospy.logwarn("Starting data loader")
    rate = rospy.Rate(10)  
    ctrlpts_sub = rospy.Subscriber("/local_planning/init_ctrlpts", ctrlpts, ctrlpts_callback)
    corridor_sub = rospy.Subscriber("/local_planning/corridor", corridor, corridor_callback)
    idx_sub = rospy.Subscriber("/local_planning/idx", idx, idx_callback)
    global rcv_flag_ctrlpts
    global rcv_flag_corridor
    global rcv_flag_idx
    while not rospy.is_shutdown():
        if rcv_flag_ctrlpts and rcv_flag_corridor and rcv_flag_idx:
            ctrlpts_input, corridors_input, idx_input = process_input_data(ctrlpts_msg, corridor_msg, idx_msg)
            rospy.loginfo(f"Received {len(ctrlpts_input)} control points")
            rospy.loginfo(f"Received {len(corridors_input)} corridor")
            rospy.loginfo(f"Received {len(idx_input)} idx")
            corridors, constraints, start_state, end_state, init_guess = process_data(ctrlpts_input, corridors_input, idx_input)
            n = len(constraints)
            if n < 5:
                rospy.loginfo("Less than 5 constraints.Failed")
            start_velocity = np.zeros(3)
            end_velocity = np.zeros(3)
            Optimizer = Iter_Optimizer(init_guess, p, n, V_max, A_max, J_max, constraints, start_state, end_state, np.zeros(3), np.zeros(3), True, 0, 0, decay_factor, 0)
            Optimizer.Optimize()
            rospy.loginfo("main : Optimized trajectory done")
            b_spline_trajectory = Optimizer.CPs_Optimizer.get_sampled_points(num_sample)
            b_spline_vel_pts, b_spline_acc_pts, b_spline_jerk_pts = Optimizer.CPs_Optimizer.get_sampled_points_derivative(num_sample)
            visualize_bspline_trajectory(b_spline_trajectory)
            collision_check = Optimizer.CPs_Optimizer.collision_check(b_spline_trajectory, corridors)
            if  collision_check[0]:
                rospy.loginfo("Collision detected")
            publish_bspline_state(b_spline_trajectory, b_spline_vel_pts, b_spline_acc_pts, b_spline_jerk_pts, Optimizer.knots_History[-1])
            rcv_flag_ctrlpts = False
            rcv_flag_corridor = False
            rcv_flag_idx = False
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

