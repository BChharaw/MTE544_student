
from mapUtilities import *
from a_star import *

POINT_PLANNER=0; TRAJECTORY_PLANNER=1

class planner:
    def __init__(self, type_, mapName="room", heuristic=EUCLIDEAN):

        self.type=type_
        self.mapName=mapName
        self.heuristic=heuristic
        self.pathCount=0

    
    def plan(self, startPose, endPose):
        
        if self.type==POINT_PLANNER:
            return self.point_planner(endPose)
        
        elif self.type==TRAJECTORY_PLANNER:
            self.costMap=None
            self.initTrajectoryPlanner()
            return self.trajectory_planner(startPose, endPose)


    def point_planner(self, endPose):
        return endPose

    def initTrajectoryPlanner(self):


        # A sigma of 0.4 m marks cells within ~0.27 m of an obstacle as walls (cost > 0.8), giving the robot clearance.
        self.m_utilites=mapManipulator(laser_sig=0.5)
            
        self.costMap=self.m_utilites.make_likelihood_field()
        

    def trajectory_planner(self, startPoseCart, endPoseCart):


        # This is to convert the cartesian coordinates into the 
        # the pixel coordinates of the map image, remmember,
        # the cost-map is in pixels. You can by the way, convert the pixels
        # to the cartesian coordinates and work by that index, the a_star finds
        # the path regardless. 
        startPose=self.m_utilites.position_2_cell(startPoseCart)
        endPose=self.m_utilites.position_2_cell(endPoseCart)
        
        # Run A* on the cost map to get the path as a list of cells.
        path = search(self.costMap, startPose, endPose, self.heuristic)

        if path is None:
            print("no path found")
            return None

        # Convert every cell on the path back into cartesian [x, y] coordinates.
        Path = list(map(self.m_utilites.cell_2_position, path))

        np.savetxt(f"path_{self.heuristic}_{self.pathCount}.csv", Path, delimiter=",", header="x,y", comments="")
        self.pathCount+=1

        # Return the path as a list of [x, y] waypoints.
        return [list(point) for point in Path]




if __name__=="__main__":

    m_utilites=mapManipulator()
    
    map_likelihood=m_utilites.make_likelihood_field()

    # you can use this part of the code to test your 
    # search algorithm regardless of the ros2 hassles
    
