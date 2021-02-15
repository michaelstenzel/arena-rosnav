#! /usr/bin/env python
import threading
from typing import Tuple

from numpy.core.numeric import normalize_axis_tuple
import rospy
import random
import numpy as np

import time  # for debuging
import threading
# observation msgs
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D, PoseStamped, PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from arena_plan_msgs.msg import RobotState,RobotStateStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
# services
from flatland_msgs.srv import StepWorld,StepWorldRequest
from std_srvs.srv import Trigger, TriggerRequest

# message filter
import message_filters

# for transformations
from tf.transformations import *

from gym import spaces
import numpy as np


class ObservationCollector():
    def __init__(self,ns: str, num_lidar_beams:int,lidar_range:float, num_humans:int=10): #
        """ a class to collect and merge observations

        Args:
            num_lidar_beams (int): [description]
            lidar_range (float): [description]
            num_humans(int): max observation number of human, default 21
        """
        self.ns = ns
        if ns is None or ns == "":
            self.ns_prefix = "/"
        else:
            self.ns_prefix = "/"+ns+"/"

        # define observation_space
        self.observation_space = ObservationCollector._stack_spaces((
            spaces.Box(low=0, high=lidar_range, shape=(num_lidar_beams,), dtype=np.float32),
            spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32) ,
            spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            spaces.Box(low=0, high=np.PINF, shape=(num_humans*2,), dtype=np.float32)
        ))

        self._scan = LaserScan()
        self._robot_pose = Pose2D()
        self._robot_vel = Twist()
        self._subgoal =  Pose2D()
        
        # topic subscriber: subgoal
        #TODO should we synchoronize it with other topics
        self._subgoal_sub = message_filters.Subscriber( f'{self.ns_prefix}subgoal', PoseStamped)#self._subgoal_sub = rospy.Subscriber("subgoal", PoseStamped, self.callback_subgoal)
        self._subgoal_sub.registerCallback(self.callback_subgoal)

        # service clients
        get_train_mode_try=0
        max_try=10
        while(get_train_mode_try<max_try):            
            try:
                self._is_train_mode = rospy.get_param("/train_mode")
                break
            except KeyError:
                get_train_mode_try+=1
                print(f'value not set retry {get_train_mode_try} times')
        # self._is_train_mode=True
        if self._is_train_mode:
            self._service_name_step=f'{self.ns_prefix}step_world' 
            self._sim_step_client = rospy.ServiceProxy(self._service_name_step, StepWorld)
        
        # message_filter subscriber: laserscan, robot_pose
        self._scan_sub = message_filters.Subscriber( f'{self.ns_prefix}{self.ns}/scan', LaserScan)
        self._robot_state_sub = message_filters.Subscriber(f'{self.ns_prefix}robot_state', RobotStateStamped)
        get_agent_topic_try=0
        max_try=10
        while(get_agent_topic_try<max_try):            
            try:
                self.human_name_str=rospy.get_param(f'{self.ns_prefix}agent_topic_string')
                break
            except KeyError:
                get_agent_topic_try+=1
                print(f'value not set retry {get_agent_topic_try} times')
        
        # self.test_topic_get=rospy.get_published_topics()
        # print(self.test_topic_get)
        # for i in range(num_obstacles):
        #     self.obstacle_name_str=self.obstacle_name_str+","+f'{self.ns_prefix}pedsim_agent_{i+1}/dynamic_human'
        # print(self.human_name_str)
        self.human_name_list=self.human_name_str.split(',')[1:]
        # print(self.human_name_list)

        # topic subscriber: different kinds of humans
        #adult
        adult_topic_list=[i for i in self.human_name_list if i.find('human')!=-1]
        # print(adult_topic_list)
        self._adult = [None]*len(adult_topic_list)
        self._adult_position, self._adult_vel= [None]*len(adult_topic_list),  [None]*len(adult_topic_list)
        # print('dynamic',adult_topic_list)
        for  i, _adult_name in enumerate(adult_topic_list):
            # print(_adult_name)
            self._adult[i] = message_filters.Subscriber(_adult_name, Odometry)
        #child

        child_topic_list=[i for i in self.human_name_list if i.find('child')!=-1]
        # print(child_topic_list)
        self._child= [None]*len(child_topic_list)
        self._child_position, self._child_vel= [None]*len(child_topic_list),  [None]*len(child_topic_list)
        for  i, _child_name in enumerate(child_topic_list):
            self._child[i] = message_filters.Subscriber(_child_name, Odometry)
        #elder
        
        elder_topic_list=[i for i in self.human_name_list if i.find('elder')!=-1]
        # print(elder_topic_list)
        self._elder= [None]*len(elder_topic_list)
        self._elder_position, self._elder_vel= [None]*len(elder_topic_list),  [None]*len(elder_topic_list)
        for  i, _elder_name in enumerate(elder_topic_list):
            self._elder[i] = message_filters.Subscriber(_elder_name, Odometry)

        # message_filters.TimeSynchronizer: call callback only when all sensor info are ready
        self.sychronized_list=[self._scan_sub, self._robot_state_sub]+self._adult+self._child+self._elder
        # print("reached here")
        self.ts = message_filters.ApproximateTimeSynchronizer(self.sychronized_list,100,slop=0.05) #,allow_headerless=True)        
        self.ts.registerCallback(self.callback_observation_received)
        # print("reached end")
    
    def get_observation_space(self):
        return self.observation_space

    def get_observations(self):
        def all_sub_received():
            ans = True
            for k, v in self._sub_flags.items():
                if v is not True:
                    ans = False
                    break
            return ans

        def reset_sub():
            self._sub_flags = dict((k, False) for k in self._sub_flags.keys())
        self._flag_all_received=False
        if self._is_train_mode: 
        # sim a step forward until all sensor msg uptodate
            i=0
            # print(self._flag_all_received)
            while(self._flag_all_received==False):
                # print(self._flag_all_received)
                self.call_service_takeSimStep()
                i+=1
        # with self._sub_flags_con:
        #     while not all_sub_received():
        #         self._sub_flags_con.wait()  # replace it with wait for later
        #     reset_sub()
        # rospy.logdebug(f"Current observation takes {i} steps for Synchronization")
        # print(f"Current observation takes {i} steps for Synchronization")
        scan = np.array(self._scan.ranges).astype(np.float32)
        rho, theta = ObservationCollector._get_goal_pose_in_robot_frame(
            self._subgoal, self._robot_pose)
        merged_obs = np.hstack([scan, np.array([rho, theta])])
        obs_dict = {}
        obs_dict["laser_scan"] = scan
        obs_dict['goal_in_robot_frame'] = [rho,theta]
        rho_a, theta_a = [None]*len(self._adult_position), [None]*len(self._adult_position)
        coordinate_a= np.empty([2,len(self._adult_position)])
        for  i, position in enumerate(self._adult_position):
            #TODO temporarily use the same fnc of _get_goal_pose_in_robot_frame
            # print("adult position",position)
            coordinate_a[0][i]=position.x
            coordinate_a[1][i]=position.y
            rho_a[i], theta_a[i] = ObservationCollector._get_goal_pose_in_robot_frame(position,self._robot_pose)
            merged_obs = np.hstack([merged_obs, np.array([rho_a[i],theta_a[i]])])
        obs_dict['adult_in_robot_frame'] = np.vstack([np.array(rho_a),np.array(theta_a)])
        obs_dict['adult_coordinates_in_robot_frame']=coordinate_a

        rho_c, theta_c = [None]*len(self._child_position), [None]*len(self._child_position)
        coordinate_c= np.empty([2,len(self._child_position)])
        for  i, position in enumerate(self._child_position):
            #TODO temporarily use the same fnc of _get_goal_pose_in_robot_frame
            coordinate_c[0][i]=position.x
            coordinate_c[1][i]=position.y
            rho_c[i], theta_c[i] = ObservationCollector._get_goal_pose_in_robot_frame(position,self._robot_pose)
            merged_obs = np.hstack([merged_obs, np.array([rho_c[i],theta_c[i]])])
        obs_dict['child_in_robot_frame'] = np.vstack([np.array(rho_c),np.array(theta_c)])
        obs_dict['child_coordinates_in_robot_frame']=coordinate_c

        rho_e, theta_e = [None]*len(self._elder_position), [None]*len(self._elder_position)
        coordinate_e= np.empty([2,len(self._elder_position)])
        for  i, position in enumerate(self._elder_position):
            #TODO temporarily use the same fnc of _get_goal_pose_in_robot_frame
            coordinate_e[0][i]=position.x
            coordinate_e[1][i]=position.y
            rho_e[i], theta_e[i] = ObservationCollector._get_goal_pose_in_robot_frame(position,self._robot_pose)
            merged_obs = np.hstack([merged_obs, np.array([rho_e[i],theta_e[i]])])
        obs_dict['elder_in_robot_frame'] = np.vstack([np.array(rho_e),np.array(theta_e)])
        obs_dict['elder_coordinates_in_robot_frame']=coordinate_e
        return merged_obs, obs_dict

    @staticmethod
    def _get_goal_pose_in_robot_frame(goal_pos: Pose2D, robot_pos: Pose2D):
        y_relative = goal_pos.y - robot_pos.y
        x_relative = goal_pos.x - robot_pos.x
        rho = (x_relative**2+y_relative**2)**0.5
        theta = (np.arctan2(y_relative, x_relative) -
                 robot_pos.theta+4*np.pi) % (2*np.pi)-np.pi
        return rho, theta

    def call_service_takeSimStep(self):
        # print("add one step")
        request = StepWorldRequest()
        try:
            response = self._sim_step_client(request)
            rospy.logdebug("step service=", response)
        except rospy.ServiceException as e:
            rospy.logdebug("step Service call failed: %s" % e)

    def callback_subgoal(self,msg_Subgoal):
        self._subgoal=self.process_subgoal_msg(msg_Subgoal)        
        return

    def callback_dynamic_obstacles(self,msg_human):
        # print("reached here callback human")
        num_adult = len(self._adult)
        num_child = len(self._child)
        num_elder = len(self._elder)
        msg_adult=msg_human[:num_adult]
        msg_child=msg_human[num_adult:num_adult+num_child]
        msg_elder=msg_human[num_adult+num_child:]
        for i,msg in enumerate(msg_adult):
            # print("x",msg.pose.pose.position.x)
            self._adult_position[i],self._adult_vel[i]=self.process_human_state_msg(msg_adult[i])
        for i,msg in enumerate(msg_child):
            self._child_position[i],self._child_vel[i]=self.process_human_state_msg(msg_child[i])
        for i,msg in enumerate(msg_elder):
            self._elder_position[i],self._elder_vel[i]=self.process_human_state_msg(msg_elder[i])
        return
        
    def callback_observation_received(self, *msg):
        # process sensor msg
        # print("reached here callback")
        self._scan=self.process_scan_msg(msg[0])
        self._robot_pose,self._robot_vel=self.process_robot_state_msg(msg[1])
        self.callback_dynamic_obstacles(msg[2:])
        # ask subgoal service
        #self._subgoal=self.call_service_askForSubgoal()
        self._flag_all_received=True
        
    def process_scan_msg(self, msg_LaserScan):
        # remove_nans_from_scan
        scan = np.array(msg_LaserScan.ranges)
        scan[np.isnan(scan)] = msg_LaserScan.range_max
        msg_LaserScan.ranges = scan
        return msg_LaserScan

    def process_robot_state_msg(self, msg_RobotStateStamped):
        state = msg_RobotStateStamped.state
        pose3d = state.pose
        twist = state.twist
        return self.pose3D_to_pose2D(pose3d), twist

    def process_human_state_msg(self,msg_humanodom):
        pose=self.process_pose_msg(msg_humanodom)
        # pose3d=state.pose
        twist=msg_humanodom.twist.twist
        return pose, twist
        
    def process_pose_msg(self,msg_PoseWithCovariance):
        # remove Covariance
        pose_with_cov=msg_PoseWithCovariance.pose
        pose=pose_with_cov.pose
        return self.pose3D_to_pose2D(pose)

    def process_subgoal_msg(self, msg_Subgoal):
        pose2d = self.pose3D_to_pose2D(msg_Subgoal.pose)
        return pose2d

    @staticmethod
    def pose3D_to_pose2D(pose3d):
        pose2d = Pose2D()
        pose2d.x = pose3d.position.x
        pose2d.y = pose3d.position.y
        quaternion = (pose3d.orientation.x, pose3d.orientation.y,
                      pose3d.orientation.z, pose3d.orientation.w)
        euler = euler_from_quaternion(quaternion)
        yaw = euler[2]
        pose2d.theta = yaw
        return pose2d

    @staticmethod
    def _stack_spaces(ss: Tuple[spaces.Box]):
        low = []
        high = []
        for space in ss:
            low.extend(space.low.tolist())
            high.extend(space.high.tolist())
        return spaces.Box(np.array(low).flatten(), np.array(high).flatten())


if __name__ == '__main__':

    rospy.init_node('states', anonymous=True)
    print("start")

    state_collector = ObservationCollector("sim_01/", 360, 10)
    i = 0
    r = rospy.Rate(100)
    while(i <= 1000):
        i = i+1
        obs = state_collector.get_observations()

        time.sleep(0.001)
