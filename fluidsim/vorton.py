'''
Created on 13/01/2012

@author: christopherd
'''
from numpy import array 
class Vorton:
    def __init__(self, vPos = array([ 0.0 , 0.0 , 0.0 ]), vVort = array([ 0.0 , 0.0 , 0.0 ]), fRadius = 0.0 ):
        self.position = vPos
        self.vorticity = vVort
        self.radius = fRadius
        self.velocity = array([ 0.0 , 0.0 , 0.0 ])
        
    def dumpSelf(self):
         return "%f %f %f %f %f %f %f %f %f %f\n" % (self.position[0], self.position[1], self.position[2], self.vorticity[0], self.vorticity[1], self.vorticity[2], self.velocity[0], self.velocity[1], self.velocity[2], self.radius)
