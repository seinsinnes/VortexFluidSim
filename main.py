import sys


import cProfile
from numpy import array
import math

from fluidsim.fluidSim import VortonSim
from fluidsim import vorticityDistribution



class GLUTVis:
    def __init__(self, viscosity , density ):
        self.fluidSim = VortonSim( viscosity , density ) #Simulation of fluid
        self.renderWindow = 0 #Identifier for render window
        self.frame = 0 #Frame counter
        self.timeNow = 0.0 #Current virtual time 
        self.initialised = False #Whether the app has been initialised
        self.mouseButtons = [0,0,0]
        self.initialiseSimConditions()
        self.timeStep    = 1.0 / 30.0
        
    
    def initialiseSimConditions(self):
        # Setup the initial conditions for the simulation
        
        self.frame = 0
        self.timeNow = 0.0

        self.fluidSim.clear()
        
        fRadius = 1.0
        fThickness = 1.0
        fMagnitude = 20.0
        numCellsPerDim = 16
        numVortonsMax = numCellsPerDim * numCellsPerDim * numCellsPerDim
        numTracersPer = 3.0

        vortons = self.fluidSim.getVortons()

        vorticityDistribution.assignVorticity( vortons , fMagnitude , numVortonsMax , vorticityDistribution.JetRing( fRadius , fThickness , array( [2.0 , 0.0 , 0.0] ) ),array([ 0.0 , 0.0 , 0.0 ]) )


        self.fluidSim.initialise( numTracersPer )
        
        #print "printing vorts"
        #for vort in vortons:
            #vort.dumpSelf()
            
        #print "dumping influence tree"
        #self.fluidSim.influenceTree.dumpSelf()

    def addVortons(self):
        fRadius = 1.0
        fThickness = 1.0
        fMagnitude = 20.0
        numCellsPerDim = 16
        numVortonsMax = numCellsPerDim * numCellsPerDim * numCellsPerDim
        vortons = self.fluidSim.getVortons()

        vorticityDistribution.assignVorticity( vortons , fMagnitude , numVortonsMax , vorticityDistribution.JetRing( fRadius , fThickness , array( [-2.0 , 0.0 , 0.0] ) ),array([ 12.0 , 0.0 , 0.0 ]) )

        
        
    ''' Move sim forward in time.
    '''
    def stepForward(self):
            
        self.fluidSim.update( self.timeStep , self.frame )

        self.frame += 1
        self.timeNow += self.timeStep
    def endSim(self):
        self.fluidSim.simWF.endWork()



if __name__ == "__main__":
    print("starting")
    glsim = GLUTVis(0.05, 1.0)
    #print glsim
    #vorts = glsim.fluidSim.getVortons()
    #for tracer in glsim.fluidSim.tracers:
        #print tracer.position, tracer.velocity
    #glsim.initialiseDisplay()

    for i in range(20):
        glsim.stepForward()
        #if i == 200:
        #    glsim.addVortons()
    glsim.endSim()
    #cProfile.runctx("glsim.fluidSim.update(1.0/60.0, 1)", globals(), locals(), "/tmp/fs.profile")
    #glsim.fluidSim.update(1.0/60.0, 1)
    #print vorts
    
