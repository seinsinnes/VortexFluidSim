'''
Created on 12/01/2012

@author: christopherd
'''
import sys
from numpy import array
import numpy
import math
import random
from multiprocessing import Process, Queue, cpu_count, Pipe

from Space.uniformGrid import UniformGrid 
from Space.nestedGrid import NestedGrid

from fluidsim.vorton import Vorton

import copy
#import functools
#import ctypes
#import computeVel
#import fluid
import Output.vtk
#import grid
import numba
from numba import cuda

OneOverFourPi       = 1.0 / (4*math.pi)
sAvoidSingularity   = math.pow( sys.float_info.min , 1.0 / 3.0 )

@cuda.jit("(float64[3],float64[3],float64[3])", inline=True,device=True)
def crossProduct(a, b, c):
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]

@cuda.jit("(float64[3],float64[3],float64[3])", inline=True,device=True)
def subtract(a, b, c):
    c[0] = a[0] - b[0]
    c[1] = a[1] - b[1]
    c[2] = a[2] - b[2]

@cuda.jit("float64(float64[3],float64[3])", inline=True,device=True)
def dotProduct(a, b):
    return(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

@cuda.jit("(float64,float64[3],float64[3])", inline=True,device=True)
def scalarProduct(a, b, c):
    c[0] = a*b[0]
    c[1] = a*b[1]
    c[2] = a*b[2]

def computeBoundaryDerivatives( jacobianGrid , vecGrid, index, dimsMinus1, reciprocalSpacing,  halfReciprocalSpacing):
    rMatrix = jacobianGrid.getCell( index)
    aIndex = array(index)
    
    if index[0] == 0:
        rMatrix[0] = ( vecGrid.getCell(aIndex + [1,0,0]) - vecGrid.getCell(index) ) * reciprocalSpacing[0]
    elif index[0] == dimsMinus1[0]:
        rMatrix[0] = ( vecGrid.getCell(index) - vecGrid.getCell(aIndex - [1,0,0] )) * reciprocalSpacing[0]
    else:
        rMatrix[0] = ( vecGrid.getCell(aIndex + [1,0,0]) - vecGrid.getCell(aIndex - [1,0,0]) ) * halfReciprocalSpacing[0]
    
    if index[1] == 0:
        rMatrix[1] = ( vecGrid.getCell(aIndex + [0,1,0]) - vecGrid.getCell( aIndex) ) * reciprocalSpacing[1]
    elif index[1] == dimsMinus1[1]:
        rMatrix[1] = ( vecGrid.getCell(aIndex) - vecGrid.getCell(aIndex - [0,1,0]) ) * reciprocalSpacing[1]
    else:
       rMatrix[1] = ( vecGrid.getCell(aIndex + [0,1,0]) - vecGrid.getCell(aIndex - [0,1,0]) ) * halfReciprocalSpacing[1]
    
    if index[2] == 0:
        rMatrix[2] = ( vecGrid.getCell(aIndex + [0,0,1]) - vecGrid.getCell(aIndex) ) * reciprocalSpacing[2]
    elif index[2] == dimsMinus1[2]:
        rMatrix[2] = ( vecGrid.getCell(aIndex) - vecGrid.getCell(aIndex - [0,0,1]) ) * reciprocalSpacing[2]
    else:
        rMatrix[2] = ( vecGrid.getCell(aIndex + [0,0,1]) - vecGrid.getCell(aIndex - [0,0,1]) ) * halfReciprocalSpacing[2]

    '''! \brief Compute velocity at a given point in space, due to influence of vortons

\param vPosition - point in space

\return velocity at vPosition, due to influence of vortons

\note This is a brute-force algorithm with time complexity O(N)
        where N is the number of vortons.  This is too slow
        for regular use but it is useful for comparisons.

'''
#@numba.jit(parallel = True)
def computeVelocityBruteForce( gPosition, vortonInfoList ):

    #numVortons          = len(self.vortons)
    vortPos = numpy.ascontiguousarray(numpy.tile(vortonInfoList[:,0:3],(gPosition.shape[0],1)))
    vortVort = numpy.ascontiguousarray(numpy.tile(vortonInfoList[:,3:6],(gPosition.shape[0],1)))
    vortRadius = numpy.ascontiguousarray(numpy.tile(vortonInfoList[:,6:9],(gPosition.shape[0],1)))
    #velocityAccumulator =  numpy.ascontiguousarray(array([0.0 , 0.0 , 0.0],dtype=numpy.float64))
    vVelocity = numpy.ascontiguousarray(numpy.tile(array([0.0 , 0.0 , 0.0],dtype=numpy.float64),(vortonInfoList.shape[0]*gPosition.shape[0],1)))
    vPosition = numpy.ascontiguousarray(numpy.repeat(gPosition,vortonInfoList.shape[0],axis=0))
    #refVel =  array([0.0 , 0.0 , 0.0])
    
    #velocityTemp = self.ctypeConv()
    #av = functools.partial(self.accumulateVelocity, vPosition)
    #vPosition = self.ctypeConv(vPosition[0],vPosition[1],vPosition[2])

    """for vortonInfo in vortonInfoList:
        #self.vortonInfoList.append([rVorton.position, rVorton.vorticity, [rVorton.radius,0.0,0.0]])
        #For each vorton...
        #vortPos = self.ctypeConv(rVorton.position[0],rVorton.position[1],rVorton.position[2])
        #vortVort = self.ctypeConv(rVorton.vorticity[0],rVorton.vorticity[1],rVorton.vorticity[2])
        #vortRadius = ctypes.c_double(rVorton.radius)
        #print "f"
        #vel = computeVel.computevel.accumulatevelocity( vPosition , vortonInfo )
        #vel = fluid.accumulateVelocity( vPosition , vortonInfo[0], vortonInfo[1], vortonInfo[2][0] )"""
    #print("shape: ",vortonInfoList[:,0:3].shape)
    accumulateVelocity( vPosition , vortPos, vortVort, vortRadius, vVelocity )
    vortonNum = len(vortonInfoList)
    print("vel_shape ", vVelocity.shape)
    velocityAccumulator = []
    for i in range(len(gPosition)):
        velocityAccumulator.append(vVelocity[0+(i*vortonNum):vortonNum+(i*vortonNum)].sum(axis=0))
    #print("velacc_shape ", velocityAccumulator.shape)
    """    #print vel
        #velocityAccumulator += vel
        #print "py"
        #refVel += self.accumulateVelocity( vPosition , vortonInfo[0], vortonInfo[1], vortonInfo[2][0] )

        #print numpy.ctypeslib.as_array(velocityTemp)
        #velocityAccumulator += numpy.ctypeslib.as_array(velocityTemp)"""
        
    #velocityAccumulator = sum([ av(vortonInfo) for vortonInfo in self.vortonInfoList])
    #return numpy.ctypeslib.as_array(velocityAccumulator)
    #print refVel
    #print velocityAccumulator
    return velocityAccumulator

@numba.guvectorize(['(float64[3],float64[3], float64[3], float64[3], float64[3])'],'(n),(n),(n),(n)->(n)', target='cuda')
def accumulateVelocity( vPosQuery , vortPos, vortVort, vortRadius, vVelocity):
    vortRadius = vortRadius[0]
    #VortonInfo = [positon, vorticity, radius]

    vNeighborToSelf = numba.cuda.local.array(3,numba.float64)
    subtract(vPosQuery, vortPos, vNeighborToSelf)                     
    radius2             = vortRadius * vortRadius                                           
    dist2               = dotProduct(vNeighborToSelf, vNeighborToSelf) + sAvoidSingularity                 
    oneOverDist         = 1 / math.sqrt( dist2 )
    
    
    ''' If the reciprocal law is used everywhere then when 2 vortices get close, they tend to jettison. '''
    '''Mitigate this by using a linear law when 2 vortices get close to each other.'''
    #print vVelocity
    if dist2 < radius2:
        # Inside vortex core
        distLaw = ( oneOverDist / radius2 )
    else:
        #Outside vortex core                             \
        distLaw = ( oneOverDist / dist2 )
    

    vortNeighborCross = numba.cuda.local.array(3,numba.float64)
    crossProduct(vortVort, vNeighborToSelf, vortNeighborCross)

    vVel = numba.cuda.local.array(3,numba.float64)
    scalarProduct(OneOverFourPi * ( 8.0 * radius2 * vortRadius ) * distLaw , vortNeighborCross, vVel)
    #print OneOverFourPi
    #vVelocity +=  OneOverFourPi * ( 8.0 * radius2 * mRadius ) * numpy.cross(mVorticity, vNeighborToSelf) * distLaw
    #vVel =  OneOverFourPi * ( 8.0 * radius2 * vortRadius ) * vortNeighborCross * distLaw 
    vVelocity[0] = vVel[0]
    vVelocity[1] = vVel[1]
    vVelocity[2] = vVel[2]
    #print
    #print vVelocity
    #return vVelocity


'''! \brief Compute Jacobian of a vector field

    \param jacobian - (output) UniformGrid of 3x3 matrix values.
                        The matrix is a vector of vectors.
                        Each component is a partial derivative with
                        respect to some direction:
                            j.a.b = d v.b / d a
                        where a and b are each one of {x,y,z}.
                        So j.x contains the partial derivatives with respect to x, etc.

    \param vec - UniformGrid of 3-vector values

'''
def computeJacobian( jacobianGrid , vecGrid ):

    spacing                 = vecGrid.getCellSpacing()
    # Avoid divide-by-zero when z size is effectively 0 (for 2D domains)
    reciprocalSpacing = array([ 1.0 / spacing[0], 1.0 / spacing[1],  0.0])
    if spacing[0] > sys.float_info.epsilon:
        reciprocalSpacing[2] = 1.0 / spacing[2]
             
    halfReciprocalSpacing = 0.5 * reciprocalSpacing
    dims = [ vecGrid.getNumPoints( 0 )   , vecGrid.getNumPoints( 1 )   , vecGrid.getNumPoints( 2 )]
    dimsMinus1 = [ vecGrid.getNumPoints( 0 )-1 , vecGrid.getNumPoints( 1 )-1 , vecGrid.getNumPoints( 2 )-1 ]
    numXY                   = dims[0] * dims[1]

    '''#define ASSIGN_Z_OFFSETS                                    \
    const unsigned offsetZM = numXY * ( index[2] - 1 ) ;    \
    const unsigned offsetZ0 = numXY *   index[2]       ;    \
    const unsigned offsetZP = numXY * ( index[2] + 1 ) ;

    #define ASSIGN_YZ_OFFSETS                                                   \
    const unsigned offsetYMZ0 = dims[ 0 ] * ( index[1] - 1 ) + offsetZ0 ;   \
    const unsigned offsetY0Z0 = dims[ 0 ] *   index[1]       + offsetZ0 ;   \
    const unsigned offsetYPZ0 = dims[ 0 ] * ( index[1] + 1 ) + offsetZ0 ;   \
    const unsigned offsetY0ZM = dims[ 0 ] *   index[1]       + offsetZM ;   \
    const unsigned offsetY0ZP = dims[ 0 ] *   index[1]       + offsetZP ;

    #define ASSIGN_XYZ_OFFSETS                                      \
    const unsigned offsetX0Y0Z0 = index[0]     + offsetY0Z0 ;   \
    const unsigned offsetXMY0Z0 = index[0] - 1 + offsetY0Z0 ;   \
    const unsigned offsetXPY0Z0 = index[0] + 1 + offsetY0Z0 ;   \
    const unsigned offsetX0YMZ0 = index[0]     + offsetYMZ0 ;   \
    const unsigned offsetX0YPZ0 = index[0]     + offsetYPZ0 ;   \
    const unsigned offsetX0Y0ZM = index[0]     + offsetY0ZM ;   \
    const unsigned offsetX0Y0ZP = index[0]     + offsetY0ZP ;'''

    # Compute derivatives for interior (i.e. away from boundaries).
    for index_z in range(1, dimsMinus1[2] ):
        #ASSIGN_Z_OFFSETS ;
        for index_y in range(1, dimsMinus1[1] ):
            #ASSIGN_YZ_OFFSETS ;
            for index_x in range( 1, dimsMinus1[0] ):
                #ASSIGN_XYZ_OFFSETS ;
                
                rMatrix = jacobianGrid.getCell([index_x,index_y,index_z])
                # Compute d/dx */
                rMatrix[0] = ( vecGrid.getCell([index_x + 1,index_y,index_z]) - vecGrid.getCell([index_x - 1,index_y,index_z]) ) * halfReciprocalSpacing[0]
                # Compute d/dy */
                rMatrix[1] = ( vecGrid.getCell([index_x,index_y + 1, index_z]) - vecGrid.getCell([index_x,index_y -1, index_z]) ) * halfReciprocalSpacing[1]
                # Compute d/dz */
                rMatrix[2] = ( vecGrid.getCell([index_x,index_y, index_z + 1]) - vecGrid.getCell([index_x,index_y, index_z - 1]) ) * halfReciprocalSpacing[2]
                jacobianGrid.setCell([index_x,index_y,index_z], rMatrix)
    ''' Compute derivatives for boundaries: 6 faces of box.
      In some situations, these macros compute extraneous data.
     A tiny bit more efficiency could be squeezed from this routine,
     but it turns out to be well under 1% of the total expense.'''

    '''#define COMPUTE_FINITE_DIFF                                                                                                             \
    Mat33 & rMatrix = jacobian[ offsetX0Y0Z0 ] ;                                                                                        \
    if( index[0] == 0 )                     { rMatrix.x = ( vec[ offsetXPY0Z0 ] - vec[ offsetX0Y0Z0 ] ) * reciprocalSpacing.x ;     }   \
    else if( index[0] == dimsMinus1[0] )    { rMatrix.x = ( vec[ offsetX0Y0Z0 ] - vec[ offsetXMY0Z0 ] ) * reciprocalSpacing.x ;     }   \
    else                                    { rMatrix.x = ( vec[ offsetXPY0Z0 ] - vec[ offsetXMY0Z0 ] ) * halfReciprocalSpacing.x ; }   \
    if( index[1] == 0 )                     { rMatrix.y = ( vec[ offsetX0YPZ0 ] - vec[ offsetX0Y0Z0 ] ) * reciprocalSpacing.y ;     }   \
    else if( index[1] == dimsMinus1[1] )    { rMatrix.y = ( vec[ offsetX0Y0Z0 ] - vec[ offsetX0YMZ0 ] ) * reciprocalSpacing.y ;     }   \
    else                                    { rMatrix.y = ( vec[ offsetX0YPZ0 ] - vec[ offsetX0YMZ0 ] ) * halfReciprocalSpacing.y ; }   \
    if( index[2] == 0 )                     { rMatrix.z = ( vec[ offsetX0Y0ZP ] - vec[ offsetX0Y0Z0 ] ) * reciprocalSpacing.z ;     }   \
    else if( index[2] == dimsMinus1[2] )    { rMatrix.z = ( vec[ offsetX0Y0Z0 ] - vec[ offsetX0Y0ZM ] ) * reciprocalSpacing.z ;     }   \
    else                                    { rMatrix.z = ( vec[ offsetX0Y0ZP ] - vec[ offsetX0Y0ZM ] ) * halfReciprocalSpacing.z ; }
    '''
    # Compute derivatives for -X boundary.
    index_x = 0
    for index_z in range(dims[2]):
        #ASSIGN_Z_OFFSETS ;
        for index_y in range(dims[1]):
            #ASSIGN_YZ_OFFSETS ;
            #ASSIGN_XYZ_OFFSETS ;
            #COMPUTE_FINITE_DIFF ;
            computeBoundaryDerivatives( jacobianGrid , vecGrid, [index_x,index_y,index_z], dimsMinus1, reciprocalSpacing,  halfReciprocalSpacing)
 

    # Compute derivatives for -Y boundary.
    index_y = 0
    for index_z in range(dims[2]):
        #ASSIGN_Z_OFFSETS ;
        #ASSIGN_YZ_OFFSETS ;
        for index_x in range(dims[0]):
            #ASSIGN_XYZ_OFFSETS ;
            computeBoundaryDerivatives( jacobianGrid , vecGrid, [index_x,index_y,index_z], dimsMinus1, reciprocalSpacing,  halfReciprocalSpacing)



    # Compute derivatives for -Z boundary.
    index_z = 0
    #ASSIGN_Z_OFFSETS ;
    for index_y in range(dims[1]):
        #ASSIGN_YZ_OFFSETS ;
        for index_x in range(dims[0]):
            #ASSIGN_XYZ_OFFSETS ;
            #COMPUTE_FINITE_DIFF ;
            computeBoundaryDerivatives( jacobianGrid , vecGrid, [index_x,index_y,index_z], dimsMinus1, reciprocalSpacing,  halfReciprocalSpacing)



    # Compute derivatives for +X boundary.
    index_x = dimsMinus1[0]
    for index_z in range(dims[2]):
        #ASSIGN_Z_OFFSETS ;
        for index_y in range(dims[1]):
            #ASSIGN_YZ_OFFSETS ;
            #ASSIGN_XYZ_OFFSETS ;
            #COMPUTE_FINITE_DIFF ;
            computeBoundaryDerivatives( jacobianGrid , vecGrid, [index_x,index_y,index_z], dimsMinus1, reciprocalSpacing,  halfReciprocalSpacing)




    # Compute derivatives for +Y boundary.
    index_y = dimsMinus1[1]
    for index_z in range( dims[2] ):
        #ASSIGN_Z_OFFSETS ;
        #ASSIGN_YZ_OFFSETS ;
        for index_x in range(dims[0]):
            #ASSIGN_XYZ_OFFSETS ;
            #COMPUTE_FINITE_DIFF ;
            computeBoundaryDerivatives( jacobianGrid , vecGrid, [index_x,index_y,index_z], dimsMinus1, reciprocalSpacing,  halfReciprocalSpacing)



    # Compute derivatives for +Z boundary.
    index_z = dimsMinus1[2]
    #ASSIGN_Z_OFFSETS ;
    for index_y in range(dims[1]):
        #ASSIGN_YZ_OFFSETS ;
        for index_x in range(dims[0]):
            #ASSIGN_XYZ_OFFSETS ;
            computeBoundaryDerivatives( jacobianGrid , vecGrid, [index_x,index_y,index_z], dimsMinus1, reciprocalSpacing,  halfReciprocalSpacing)


#undef COMPUTE_FINITE_DIFF
#undef ASSIGN_XYZ_OFFSETS
#undef ASSIGN_YZ_OFFSETS
#undef ASSIGN_Z_OFFSETS






''' \brief Basic particle for use in a visual effects particle system
'''
class Particle:
        ''' \brief Construct a particle
        '''
        def ___init___(self):
            self.position = array([0.0 , 0.0 , 0.0]) #///< Position (in world units) of center of particle
            self.velocity = array([0.0 , 0.0 , 0.0]) #///< Velocity of particle
            self.orientation = array([0.0 , 0.0 , 0.0]) #///< Orientation of particle, in axis-angle form where angle=|orientation|
            self.angularVelocity = array([0.0 , 0.0 , 0.0]) #///< Angular velocity of particle
            self.mass = 0.0 #///< Mass of particle
            self.size = 0.0 #///< Size of particle
            self.birthTime =  0 #///< Birth time of particle, in "ticks"
            
class SimWorkers:
    def __init__(self):
        self.resultQ = Queue()
        self.simWorkers = []
        self.workerComms =[]
        self.numWorkers = 8
        self.workersDoneCount = 0
        for i in range(self.numWorkers):
            parent_conn,child_conn = Pipe()
            self.simWorkers.append(Process(target=self.simWork, args=(child_conn,)))
            self.workerComms.append(parent_conn)
            #child_conn.close()
        for worker in self.simWorkers:
                worker.start()
            
    def broadcastVortons(self, vortonList):
        print("broadcast vlist")
        for c in self.workerComms:
            c.send(vortonList)
        print ("bc done")
        
    def scatterWork(self, workList):
        num = len(workList)
        sliceBegin = 0
        sliceEnd = 0
        workerNum = 0
        divr = self.numWorkers
        print("distWork" + str(num))
        while num != 0:
            d = int(num/divr)
            num = num - d
            divr -= 1
            sliceEnd = sliceBegin + d
            #print sliceEnd,sliceBegin
            self.workerComms[workerNum].send(workList[sliceBegin:sliceEnd])
            sliceBegin += d
            workerNum +=1
            
    def simWork(self, dpipe):
        while True:
            #for c in self.workerComms:
            #    c.close()
            print("worker ready")
            vortonList = dpipe.recv()
            print ("recv'd")
            workList = dpipe.recv()
            print ("worklist recv'd")
            if vortonList is None or workList is None:
                break
            #computeVel.computevel.setvortoninfolist(vortonList, len(vortonList))
            #print "vort"            
            #print len(vortonList),len(vortonList[0]),len(vortonList[0][0]),vortonList[0][0].dtype
            #fluid.setVortonStatsList(vortonList)
            #self.vortonInfoList = vortonList
            for pos in workList:
                #print pos[1]
                #vel = computeVel.computevel.computevelocitybruteforce(pos[0])
                vel = computeVelocityBruteForce(pos[0], vortonList)
                #vel = fluid.computeVelocityAtPosition(pos[0])
                #if vel[0]-vel2[0]!= 0 or vel[1]-vel2[1]!= 0 or vel[2]-vel2[2]!= 0:
                #    print "decrp: ",vel,vel2,pos[0]
                self.resultQ.put([pos[1],vel])
            
            self.resultQ.put(None)

    def endWork(self):
        for workerNum in range(self.numWorkers):
            self.workerComms[workerNum].send(None)
            self.workerComms[workerNum].send(None)
        for workerNum in range(self.numWorkers):
            self.simWorkers[workerNum].join()
            
    def getResult(self):
        result = None 
        while result == None:
            result = self.resultQ.get()
            if result == None:
                self.workersDoneCount +=1
            if self.workersDoneCount == self.numWorkers:
                print("All complete")
                self.workersDoneCount = 0
                break
        #print result
        return result
            #get Q item
            #process
            #put result on to Q

class VortonSim:
    def __init__(self, viscosity , density):
        maxFloatValue = sys.float_info.max 
        
        self.minCorner = array([maxFloatValue, maxFloatValue, maxFloatValue])
        self.maxCorner = -self.minCorner
        self.viscosity = viscosity
        #self.velocityGrid =
        self.influenceTree = NestedGrid()   # Influence tree
        self.velGrid = UniformGrid()
        self.circulationInitial = array([ 0.0 , 0.0 , 0.0 ])
        self.linearImpulseInitial = array( [0.0 , 0.0 , 0.0 ] )
        self.averageVorticity = array( [0.0 , 0.0 , 0.0 ])
        self.fluidDensity = density
        self.massPerParticle = 0.0
        self.vortons = []
        self.tracers = []
        #self.simWF = SimWorkers()
        
        #self.computeVel = ctypes.CDLL("/home/users/christopherd/workspace/fluidsim_py/computeVel.so")
        #self.ctypeConv = ctypes.c_double * 3

    def clear(self):
        self.vortons = []
            #mInfluenceTree.Clear() ;
        #self.velocityGrid.Clear()
        self.tracers = []
    
    def getVortons(self):
        return self.vortons
            
        
    
    ''' \brief Compute the total circulation and linear impulse of all vortons in this simulation.

    \param vCirculation - Total circulation, the volume integral of vorticity, computed by this routine.

    \param vLinearImpulse - Volume integral of circulation weighted by position, computed by this routine.

    '''
    def conservedQuantities( self):

        # Zero accumulators.
        vCirculation = vLinearImpulse = array([ 0.0 , 0.0 , 0.0 ])
        for rVorton in self.vortons:
            #For each vorton in this simulation...
            volumeElement   = math.pow( rVorton.radius, 3 ) * 8.0
            #Accumulate total circulation.
            vCirculation    += rVorton.vorticity * volumeElement
            #Accumulate total linear impulse.
            vLinearImpulse  += numpy.cross(rVorton.position, rVorton.vorticity) * volumeElement

        return vCirculation, vLinearImpulse
    
    ''' \brief Compute the average vorticity of all vortons in this simulation.

    \note This is used to compute a hacky, non-physical approximation to
            viscous vortex diffusion.

    '''
    def computeAverageVorticity(self):
        self.averageVorticity = array([0.0, 0.0, 0.0])
        for rVorton in self.vortons:
            self.averageVorticity += rVorton.vorticity
        
        self.averageVorticity = self.averageVorticity / float(len(self.vortons))
    
    ''' \brief Initialize a vortex particle fluid simulation

    \note This method assumes the vortons have been initialized.
            That includes removing any vortons embedded inside
            rigid bodies.
    '''
    def initialise(self, numTracersPerCellCubeRoot):
        self.circulationInitial , self.linearImpulseInitial = self.conservedQuantities()
        self.computeAverageVorticity()
        self.createInfluenceTree() # This is a marginally superfluous call.  We only need the grid geometry to seed passive tracer particles.
        self.initializePassiveTracers( numTracersPerCellCubeRoot )


        domainVolume = self.influenceTree[0].getExtent()[0] * self.influenceTree[0].getExtent()[1] * self.influenceTree[0].getExtent()[2]
        if 0.0 == self.influenceTree[0].getExtent()[2]:
        # Domain is 2D in XY plane.
            domainVolume = self.influenceTree[0].getExtent()[0] * self.influenceTree[0].getExtent()[1]

        totalMass = domainVolume * self.fluidDensity
        numTracersPerCell = math.pow( numTracersPerCellCubeRoot, 3 )
        self.massPerParticle = totalMass / float( self.influenceTree[0].getGridCapacity() * numTracersPerCell )

    '''! \brief Find axis-aligned bounding box for all vortons in this simulation.
    '''
    def findBoundingBox( self ):
        #QUERY_PERFORMANCE_ENTER ;
        floatMax = sys.float_info.max
        self.minCorner = array([floatMax, floatMax, floatMax])
        self.maxCorner = -self.minCorner 
        #print "finding"
        for rVorton in self.vortons:
            # For each vorton in this simulation...
            #Find corners of axis-aligned bounding box.
            #print rVorton.position
            self.minCorner[0] = min(self.minCorner[0], rVorton.position[0])
            self.minCorner[1] = min(self.minCorner[1], rVorton.position[1])
            self.minCorner[2] = min(self.minCorner[2], rVorton.position[2])
            self.maxCorner[0] = max(self.maxCorner[0], rVorton.position[0])
            self.maxCorner[1] = max(self.maxCorner[1], rVorton.position[1])
            self.maxCorner[2] = max(self.maxCorner[2], rVorton.position[2])
        #QUERY_PERFORMANCE_EXIT( VortonSim_CreateInfluenceTree_FindBoundingBox_Vortons ) ;

        #QUERY_PERFORMANCE_ENTER ;
        
        for rTracer in self.tracers:
            # For each passive tracer particle in this simulation...
            #Find corners of axis-aligned bounding box.
            self.minCorner[0] = min(self.minCorner[0], rTracer.position[0])
            self.minCorner[1] = min(self.minCorner[1], rTracer.position[1])
            self.minCorner[2] = min(self.minCorner[2], rTracer.position[2])
            self.maxCorner[0] = max(self.maxCorner[0], rTracer.position[0])
            self.maxCorner[1] = max(self.maxCorner[1], rTracer.position[1])
            self.maxCorner[2] = max(self.maxCorner[2], rTracer.position[2])
            #UpdateBoundingBox( mMinCorner , mMaxCorner , rTracer.mPosition ) ;
    
            #QUERY_PERFORMANCE_EXIT( VortonSim_CreateInfluenceTree_FindBoundingBox_Tracers ) ;

        #Slightly enlarge bounding box to allow for round-off errors.
        extent = self.maxCorner - self.minCorner
        nudge =  extent * sys.float_info.epsilon
        self.minCorner -= nudge
        self.maxCorner += nudge
        #print "stopped"
    ''' \brief Create nested grid vorticity influence tree

    Each layer of this tree represents a simplified, aggregated version of
    all of the information in its "child" layer, where each
    "child" has higher resolution than its "parent".

    \see MakeBaseVortonGrid, AggregateClusters

    Derivation:

    Using conservation properties, I_0 = I_0' , I_1 = I_1' , I_2 = I_2'

    I_0 : wx d = w1x d1 + w2x d2
        : wy d = w1y d1 + w2y d2
        : wz d = w1z d1 + w2z d2

        These 3 are not linearly independent:
    I_1 : ( y wz - z wy ) d = ( y1 wz1 - z1 wy1 ) d1 + ( y2 wz2 - z2 wy2 ) d2
        : ( z wx - x wz ) d = ( z1 wx1 - x1 wz1 ) d1 + ( z2 wx2 - x2 wz2 ) d2
        : ( x wy - y wx ) d = ( x1 wy1 - y1 wx1 ) d1 + ( x2 wy2 - y2 wx2 ) d2

    I_2 : ( x^2 + y^2 + z^2 ) wx d = (x1^2 + y1^2 + z1^2 ) wx1 d1 + ( x2^2 + y2^2 + z2^2 ) wx2 d2
        : ( x^2 + y^2 + z^2 ) wy d = (x1^2 + y1^2 + z1^2 ) wy1 d1 + ( x2^2 + y2^2 + z2^2 ) wy2 d2
        : ( x^2 + y^2 + z^2 ) wz d = (x1^2 + y1^2 + z1^2 ) wz1 d1 + ( x2^2 + y2^2 + z2^2 ) wz2 d2

    Can replace I_2 with its magnitude:
              ( x^2  + y^2  + z^2  ) ( wx^2  + wy^2  + wz^2  )^(1/2) d
            = ( x1^2 + y1^2 + z1^2 ) ( wx1^2 + w1y^2 + w1z^2 )^(1/2) d1
            + ( x2^2 + y2^2 + z2^2 ) ( wx2^2 + w2y^2 + w2z^2 )^(1/2) d2

    '''
    def createInfluenceTree( self):
    
        #QUERY_PERFORMANCE_ENTER ;
        self.findBoundingBox() # Find axis-aligned bounding box that encloses all vortons.
        #QUERY_PERFORMANCE_EXIT( VortonSim_CreateInfluenceTree_FindBoundingBox ) ;

        #Create skeletal nested grid for influence tree.
        numVortons = len(self.vortons)
        #print "min"
        #print self.minCorner
        #print self.maxCorner
        #print "create ugSkel"
        ugSkeleton = UniformGrid()  # Uniform grid with the same size & shape as the one holding aggregated information about mVortons.
        ugSkeleton.defineShape( numVortons , self.minCorner , self.maxCorner , True )
        templateVorton = Vorton()
        ugSkeleton.init(templateVorton)
        #print "skelcont: "
        #print len(ugSkeleton.contents)
        self.influenceTree.initialise( ugSkeleton ) # Create skeleton of influence tree.

        #QUERY_PERFORMANCE_ENTER ;
        self.makeBaseVortonGrid()
        #QUERY_PERFORMANCE_EXIT( VortonSim_CreateInfluenceTree_MakeBaseVortonGrid ) ;

        #QUERY_PERFORMANCE_ENTER ;
        numLayers = self.influenceTree.getDepth()
        for uParentLayer in range( 1, numLayers ):
        # For each layer in the influence tree...
            self.aggregateClusters( uParentLayer )
    
        #QUERY_PERFORMANCE_EXIT( VortonSim_CreateInfluenceTree_AggregateClusters ) ;


    ''' \brief Create base layer of vorton influence tree.

    This is the leaf layer, where each grid cell corresponds (on average) to
    a single vorton.  Some cells might contain multiple vortons and some zero.
    Each cell effectively has a single "supervorton" which its parent layers
    in the influence tree will in turn aggregate.

    \note This implementation of gridifying the base layer is NOT suitable
            for Eulerian operations like approximating spatial derivatives
            of vorticity or solving a vector Poisson equation, because this
            routine associates each vortex with a single corner point of the
            grid cell that contains it.  To create a grid for Eulerian calculations,
            each vorton would contribute to all 8 corner points of the grid
            cell that contains it.

            We could rewrite this to suit "Eulerian" operations, in which case
            we would want to omit "size" and "position" since the grid would
            implicitly represent that information.  That concern goes hand-in-hand
            with the method used to compute velocity from vorticity.
            Ultimately we need to make sure theoretically conserved quantities behave as expected.

    \note This method assumes the influence tree skeleton has already been created,
            and the leaf layer initialized to all "zeros", meaning it contains no
            vortons.

    '''
    def makeBaseVortonGrid(self):

        numVortons = len(self.vortons)

        #ugAux = UniformGrid( self.influenceTree[0] ) # Temporary auxilliary information used during aggregation.
        ugAux = copy.copy(self.influenceTree[0])
        #print "mBVG"
        ugAux.init()

        # Compute preliminary vorticity grid.
        #for( unsigned uVorton = 0 ; uVorton < numVortons ; ++ uVorton )
        for rVorton in self.vortons:
            #For each vorton in this simulation...
            position   = rVorton.position
            #uOffset     = self.influenceTree[0].OffsetOfPosition( rPosition )
            indices = self.influenceTree[0].indicesOfPosition(position)
            #print "pts: %d %d %d" % (self.influenceTree[0].getNumPoints(0), self.influenceTree[0].getNumPoints(1), self.influenceTree[0].getNumPoints(2))

            vortonCell = self.influenceTree[0].getCell(indices)
            vortonAux  = ugAux.getCell(indices)
            vortMag     = math.sqrt(numpy.dot(rVorton.vorticity, rVorton.vorticity))

            #print vortMag, rVorton.position
            vortonCell.position  += rVorton.position * vortMag # Compute weighted position -- to be normalized later.
            
            vortonCell.vorticity += rVorton.vorticity          # Tally vorticity sum.
            vortonCell.radius     = rVorton.radius             # Assign volume element size.
            #  OBSOLETE. See comments below: UpdateBoundingBox( rVortonAux.mMinCorner , rVortonAux.mMaxCorner , rVorton.mPosition ) ;
            self.influenceTree[0].setCell(indices, vortonCell)
            
            vortonAux+= vortMag
            #print vortonCell.position, vortonCell.vorticity, vortonAux, indices
            #print vortMag
            ugAux.setCell(indices, vortonAux)
            
        # Post-process preliminary grid; normalize center-of-vorticity and compute sizes, for each grid cell.
        num = [   self.influenceTree[0].getNumPoints( 0 ) , \
                                self.influenceTree[0].getNumPoints( 1 ) , \
                                self.influenceTree[0].getNumPoints( 2 ) ]
        numXY = num[0] * num[1]
            
        for idx_z in range(num[2]):
            for idx_y in range(num[1]):
                for idx_x in range(num[0]):
                    idx_indices = [idx_x,idx_y,idx_z]
                    vortonAux  = ugAux.getCell(idx_indices)
                    if vortonAux != sys.float_info.min:
                        # This cell contains at least one vorton.
                        vortonCell = self.influenceTree[0].getCell(idx_indices)
                            
                        # Normalize weighted position sum to obtain center-of-vorticity.
                        vortonCell.position /= vortonAux
                        #print vortonAux
                        self.influenceTree[0].setCell(idx_indices, vortonCell)
                        #print vortonCell.position
                            
                            
    ''' \brief Aggregate vorton clusters from a child layer into a parent layer of the influence tree

    This routine assumes the given parent layer is empty and its child layer (i.e. the layer
    with index uParentLayer-1) is populated.

    \param uParentLayer - index of parent layer into which aggregated influence information will be stored.
    This must be greater than 0 because the base layer, which has no children, has index 0.

    \see CreateInfluenceTree

    '''
                            
    def aggregateClusters(self, uParentLayer ):

        rParentLayer  = self.influenceTree[ uParentLayer ]

        # number of cells in each grid cluster
        pClusterDims = self.influenceTree.getDecimations( uParentLayer )
        #print pClusterDims
        numCells = [ rParentLayer.getNumCells( 0 ) , rParentLayer.getNumCells( 1 ) , rParentLayer.getNumCells( 2 ) ]
        #numXY               = rParentLayer.GetNumPoints( 0 ) * rParentLayer.GetNumPoints( 1 ) ;
        # (Since this loop writes to each parent cell, it should readily parallelize without contention.)
        #unsigned idxParent[3] ;
        for  idxParent_z in range( numCells[2] ):
            #const unsigned offsetZ = idxParent[2] * numXY ;
            for idxParent_y in range( numCells[1] ):
                #const unsigned offsetYZ = idxParent[1] * rParentLayer.GetNumPoints( 0 ) + offsetZ ;
                for idxParent_x in range( numCells[0] ):
                    # For each cell in the parent layer...
                    #const unsigned offsetXYZ = idxParent[0] + offsetYZ ;
                    rChildLayer   = self.influenceTree[ uParentLayer - 1 ]
                    parentIdx = [idxParent_x, idxParent_y, idxParent_z]
                    vortonParent = rParentLayer.getCell(parentIdx)
                    #VortonClusterAux vortAux ;
                    vortNormSum = sys.float_info.min 
                    #unsigned clusterMinIndices[ 3 ] ;
                    clusterMinIndices = self.influenceTree.getChildClusterMinCornerIndex( pClusterDims , [idxParent_x, idxParent_y, idxParent_z ])
                    #print clusterMinIndices
                    #increment = [ 0 , 0 , 0 ]
                    numXchild  = rChildLayer.getNumPoints( 0 )
                    numXYchild = numXchild * rChildLayer.getNumPoints( 1 )
                    # For each cell of child layer in this grid cluster...
                 
                    for increment_z in range( int(pClusterDims[2]) ):
                
                        offsetZ = ( clusterMinIndices[2] + increment_z ) * numXYchild ;
                        for increment_y in range( int(pClusterDims[1]) ):

                            offsetYZ = ( clusterMinIndices[1] + increment_y ) * numXchild + offsetZ ;
                            for increment_x in range( int(pClusterDims[0]) ):
                        
                                offsetXYZ       = ( clusterMinIndices[0] + increment_x ) + offsetYZ ;
                                vortonChild    = rChildLayer.getCell([ clusterMinIndices[0] + increment_x, clusterMinIndices[1] + increment_y, clusterMinIndices[2] + increment_z])
                                vortMag         = math.sqrt(numpy.dot(vortonChild.vorticity, vortonChild.vorticity))

                                #Aggregate vorton cluster from child layer into parent layer:
                                vortonParent.position  += vortonChild.position * vortMag
                                vortonParent.vorticity += vortonChild.vorticity
                                vortNormSum     += vortMag
                                if vortonChild.radius != 0.0:
                                    vortonParent.radius  = vortonChild.radius
  
                    # Normalize weighted position sum to obtain center-of-vorticity.
                    # (See analogous code in MakeBaseVortonGrid.)
                    vortonParent.position /= vortNormSum
                    rParentLayer.setCell( parentIdx, vortonParent )

    '''! \brief Initialize passive tracers

    \note This method assumes the influence tree skeleton has already been created,
            and the leaf layer initialized to all "zeros", meaning it contains no
            vortons.
    '''
    def initializePassiveTracers( self, multiplier ):

        vSpacing        = self.influenceTree[0].getCellSpacing()
        # Must keep tracers away from maximal boundary by at least cell.  Note the +vHalfSpacing in loop.
        begin        = [ 1*self.influenceTree[0].getNumCells(0)/8 , 1*self.influenceTree[0].getNumCells(1)/8 , 1*self.influenceTree[0].getNumCells(2)/8 ]
        end          = [ 7*self.influenceTree[0].getNumCells(0)/8 , 7*self.influenceTree[0].getNumCells(1)/8 , 7*self.influenceTree[0].getNumCells(2)/8 ]
        pclSize      = 2.0 * math.pow( vSpacing[0] * vSpacing[1] * vSpacing[2] , 2.0 / 3.0 ) / float( multiplier )
        noise           = vSpacing / float( multiplier )
        print ("n")
        print (noise)
    #unsigned        idx[3]          ;

        nt           = [ multiplier , multiplier , multiplier ]
        #print "trace",begin,end
        for idx_z in range(int(begin[2]),int(end[2]+1)):
            for idx_y in range(int(begin[1]), int(end[1]+1)):
                for idx_x in range(int(begin[0]), int(end[0]+1)):
                    
                    #For each interior grid cell...
                    #Vec3 vPosMinCorner ;
                    vPosMinCorner = self.influenceTree[0].positionFromIndices( [idx_x, idx_y, idx_z] )
                    #print vPosMinCorner


                    for it_z in range(int(nt[2])):
                        for it_y in range( int(nt[1]) ):
                            for it_x in range( int(nt[0]) ):
                                pcl = Particle()
                                pcl.velocity            = array([ 0.0 , 0.0 , 0.0 ])
                                pcl.orientation        = array( [0.0 , 0.0 , 0.0 ])
                                pcl.angularVelocity    = array( [0.0 , 0.0 , 0.0 ])
                                pcl.mass               = 1.0
                                pcl.size                = pclSize
                                pcl.birthTime          = 0
                    
                                vShift = array([ float( it_x ) / float( nt[0] ) * vSpacing[0] , \
                                                float( it_y ) / float( nt[1] ) * vSpacing[1] , \
                                                float( it_z ) / float( nt[2] ) * vSpacing[2] ])
                                randNoise = [(random.random() - 0.5), (random.random() - 0.5), (random.random() - 0.5)]* noise
                                #print "rn"
                                #print randNoise
                                pcl.position           = vPosMinCorner + vShift + randNoise
                                self.tracers.append(pcl)
                                
    


    '''! \brief Compute velocity due to vortons, for a subset of points in a uniform grid

    \param izStart - starting value for z index

    \param izEnd - ending value for z index

    \see CreateInfluenceTree, ComputeVelocityGrid

    \note This routine assumes CreateInfluenceTree has already executed,
            and that the velocity grid has been allocated.

    '''
    def computeVelocityGridSlice(self, izStart , izEnd ):

        '''#if VELOCITY_FROM_TREE
        const size_t        numLayers   = mInfluenceTree.GetDepth() ;
        #endif'''
        #largestVel = 0.0
        self.vortonInfoList = []
        self.vortonInfoList2 = []
        count = 0
        for rVorton in self.vortons:
            #self.vortonInfoList.append([rVorton.position, rVorton.vorticity, [rVorton.radius,0.0,0.0]])
            #For each vorton...
            #vortPos = self.ctypeConv(rVorton.position[0],rVorton.position[1],rVorton.position[2])
            #vortVort = self.ctypeConv(rVorton.vorticity[0],rVorton.vorticity[1],rVorton.vorticity[2])
            #vortRadius = ctypes.c_double(rVorton.radius)
            #self.vortonInfoList.append([rVorton.position, rVorton.vorticity, [rVorton.radius, 0.0, 0.0]])
            self.vortonInfoList2.append(numpy.concatenate([rVorton.position, rVorton.vorticity,numpy.array([rVorton.radius,0,0])]))
            #print count,self.vortonInfoList[-1]
            count +=1
        self.vortonInfoList2 = numpy.array(self.vortonInfoList2,dtype=numpy.float64)
        #f = open("/tmp/velocity.dmp", "w")
        #self.simWF.broadcastVortons(self.vortonInfoList)
        #self.simWF.broadcastVortons(numpy.array(self.vortonInfoList2))
        #computeVel.computevel.setvortoninfolist(self.vortonInfoList, len(self.vortonInfoList))
        vMinCorner  = self.velGrid.getMinCorner()
        nudge       = 1.0 - 2.0 * sys.float_info.epsilon
        vSpacing    = self.velGrid.getCellSpacing() * nudge
        dims     =   [ self.velGrid.getNumPoints( 0 ) \
                       , self.velGrid.getNumPoints( 1 ) \
                       , self.velGrid.getNumPoints( 2 ) ]
        
        numXY       = dims[0] * dims[1]
        gPoints = []
        for idx_z in range(izStart, izEnd ):
            #For subset of z index values...
            vPosition = array([0.0,0.0,0.0], dtype = numpy.float32)
            #Compute the z-coordinate of the world-space position of this gridpoint.
            vPosition[2] = vMinCorner[2] + float( idx_z ) * vSpacing[2]
            #Precompute the z contribution to the offset into the velocity grid.
            offsetZ = idx_z * numXY
            for idx_y in range(0, dims[1]):
                #For every gridpoint along the y-axis...
                #Compute the y-coordinate of the world-space position of this gridpoint.
                vPosition[1] = vMinCorner[1] + float( idx_y ) * vSpacing[1]
                # Precompute the y contribution to the offset into the velocity grid.
                offsetYZ = idx_y * dims[0] + offsetZ
                for idx_x in range( 0, dims[0] ):
                    # For every gridpoint along the x-axis...
                    #Compute the x-coordinate of the world-space position of this gridpoint.
                    vPosition[0] = vMinCorner[0] + float( idx_x ) * vSpacing[0]
                    # Compute the offset into the velocity grid.
                    offsetXYZ = idx_x + offsetYZ
                    
                    #Compute the fluid flow velocity at this gridpoint, due to all vortons.
                    '''#if VELOCITY_FROM_TREE
                    static const unsigned zeros[3] = { 0 , 0 , 0 } # Starter indices for recursive algorithm
                    mVelGrid[ offsetXYZ ] = ComputeVelocity( vPosition , zeros , numLayers - 1  ) ;
                    #else'''   # Slow accurate dirrect summation algorithm

                    
                    #f.write( str(self.velGrid.getCell([idx_x,idx_y,idx_z])[0]) +"\t" + str(self.velGrid.getCell([idx_x,idx_y,idx_z])[1]) + "\t" + str(self.velGrid.getCell([idx_x,idx_y,idx_z])[2]))
                    #f.write(str(vPosition[0])+"\t"+str(vPosition[1])+"\t"+str(vPosition[2])+"\n")
                    #velMag = vel[0]**2 + vel[1]**2 + vel[2]**2
                    #if velMag > largestVel:
                    #largestVel = velMag
                    #print vel
                    #vel2 = self.computeVelocityBruteForce( vPosition )
                    #print [idx_x,idx_y,idx_z]
                    #self.simWF.workQ.put([vPosition,[idx_x,idx_y,idx_z]])
                    
                    gPoints.append(array(vPosition))
                    #vVelocity = computeVelocityBruteForce(array(vPosition,dtype=numpy.float64), self.vortonInfoList2)
                    #self.velGrid.setCell([idx_x,idx_y,idx_z], vVelocity)
                    #vel = computeVel.computevel.computevelocitybruteforce( vPosition )
                    #print "f"
                    #print vel
                    #print "py"
                    #print vel2
                    #self.velGrid.setCell([idx_x,idx_y,idx_z], vel)
        #self.simWF.workQ.put(0)
        #self.simWF.scatterWork(gPoints)
        """f2 = open("/tmp/grid.b","w")
        
        
        zi = 0
        for gz in self.velGrid.contents:
            yi = 0
            for gy in gz:
                xi = 0
                for gx in gy:
                    f2.write(str([zi,yi,xi]) + " ")
                    f2.write(str(gx) + "\n")
                    xi += 1
                yi+=1
            zi+=1
        f2.close()"""
        print("getting results")
        vVelocity = computeVelocityBruteForce(array(gPoints,dtype=numpy.float64), self.vortonInfoList2)
        #while True:
        #    result = self.simWF.getResult()
        #    if result == None:
        #        break
        
        #    self.velGrid.setCell(result[0], result[1])
        print("results recv'd")
        i = 0
        for idx_z in range(izStart, izEnd ):
            #For subset of z index values...
            vPosition = array([0.0,0.0,0.0], dtype = numpy.float32)
            #Compute the z-coordinate of the world-space position of this gridpoint.
            vPosition[2] = vMinCorner[2] + float( idx_z ) * vSpacing[2]
            #Precompute the z contribution to the offset into the velocity grid.
            offsetZ = idx_z * numXY
            for idx_y in range(0, dims[1]):
                #For every gridpoint along the y-axis...
                #Compute the y-coordinate of the world-space position of this gridpoint.
                vPosition[1] = vMinCorner[1] + float( idx_y ) * vSpacing[1]
                # Precompute the y contribution to the offset into the velocity grid.
                offsetYZ = idx_y * dims[0] + offsetZ
                for idx_x in range( 0, dims[0] ):
                    # For every gridpoint along the x-axis...
                    #Compute the x-coordinate of the world-space position of this gridpoint.
                    vPosition[0] = vMinCorner[0] + float( idx_x ) * vSpacing[0]
                    # Compute the offset into the velocity grid.
                    offsetXYZ = idx_x + offsetYZ
                    self.velGrid.setCell([idx_x,idx_y,idx_z], vVelocity[i])
                    i +=1

        """f2 = open("/tmp/grid.a","w")
        
        
        zi = 0
        for gz in self.velGrid.contents:
            yi = 0
            for gy in gz:
                xi = 0
                for gx in gy:
                    f2.write(str([zi,yi,xi]) + " ")
                    f2.write(str(gx) + "\n")
                    xi += 1
                yi+=1
            zi+=1
        f2.close()"""
    '''! \brief Compute velocity at a given point in space, due to influence of vortons

    \param vPosition - point in space

    \return velocity at vPosition, due to influence of vortons

    \note This is a brute-force algorithm with time complexity O(N)
            where N is the number of vortons.  This is too slow
            for regular use but it is useful for comparisons.

    '''
    def computeVelocityBruteForce(self, vPosition ):

        #numVortons          = len(self.vortons)
        velocityAccumulator =  array([0.0 , 0.0 , 0.0])
        refVel =  array([0.0 , 0.0 , 0.0])
        
        #velocityTemp = self.ctypeConv()
        #av = functools.partial(self.accumulateVelocity, vPosition)
        #vPosition = self.ctypeConv(vPosition[0],vPosition[1],vPosition[2])

        for vortonInfo in self.vortonInfoList:
            #self.vortonInfoList.append([rVorton.position, rVorton.vorticity, [rVorton.radius,0.0,0.0]])
            #For each vorton...
            #vortPos = self.ctypeConv(rVorton.position[0],rVorton.position[1],rVorton.position[2])
            #vortVort = self.ctypeConv(rVorton.vorticity[0],rVorton.vorticity[1],rVorton.vorticity[2])
            #vortRadius = ctypes.c_double(rVorton.radius)
            #print "f"
            #vel = computeVel.computevel.accumulatevelocity( vPosition , vortonInfo )
            vel += fluid.accumulateVelocity( vPosition , vortonInfo[0], vortonInfo[1], vortonInfo[2][0] )
            #print vel
            velocityAccumulator += vel
            #print "py"
            #refVel += self.accumulateVelocity( vPosition , vortonInfo[0], vortonInfo[1], vortonInfo[2][0] )

            #print numpy.ctypeslib.as_array(velocityTemp)
            #velocityAccumulator += numpy.ctypeslib.as_array(velocityTemp)
            
        #velocityAccumulator = sum([ av(vortonInfo) for vortonInfo in self.vortonInfoList])
        #return numpy.ctypeslib.as_array(velocityAccumulator)
        #print refVel
        #print velocityAccumulator
        return velocityAccumulator

    ''' \brief Compute velocity due to vortons, for every point in a uniform grid

    \see CreateInfluenceTree

    \note This routine assumes CreateInfluenceTree has already executed.

    '''
    def computeVelocityGrid( self ):

        #self.velGrid.clear() ;                                  # Clear any stale velocity information
        self.velGrid.copyShape( self.influenceTree[0] )             # Use same shape as base vorticity grid. (Note: could differ if you want.)
        self.velGrid.init()                                   # Reserve memory for velocity grid.

        numZ = self.velGrid.getNumPoints( 2 )

        '''#if USE_TBB
        // Estimate grain size based on size of problem and number of processors.
        const unsigned grainSize =  MAX2( 1 , numZ / gNumberOfProcessors ) ;
        // Compute velocity grid using multiple threads.
        parallel_for( tbb::blocked_range<size_t>( 0 , numZ , grainSize ) , VortonSim_ComputeVelocityGrid_TBB( this ) ) ;
        #else'''
        self.computeVelocityGridSlice( 0 , numZ )



    '''\brief Stretch and tilt vortons using velocity field

    \param timeStep - amount of time by which to advance simulation

    \param uFrame - frame counter

    \see AdvectVortons

    \see J. T. Beale, A convergent three-dimensional vortex method with
            grid-free stretching, Math. Comp. 46 (1986), 401-24, April.

    \note This routine assumes CreateInfluenceTree has already executed.

    '''
    def stretchAndTiltVortons(self, timeStep , uFrame ):
        # Compute all gradients of all components of velocity.
        velocityJacobianGrid = UniformGrid()
        velocityJacobianGrid.copyShape( self.velGrid )
        velocityJacobianGrid.init(numpy.matrix([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]))
        computeJacobian( velocityJacobianGrid , self.velGrid )

        if  0.0 == self.velGrid.getExtent()[0] \
        or  0.0 == self.velGrid.getExtent()[1] \
        or  0.0 == self.velGrid.getExtent()[2]:
            # Domain is 2D, so stretching & tilting does not occur.
            return
        

        #numVortons = mVortons.Size()

        for rVorton in self.vortons:
        # For each vorton...
            velJac = velocityJacobianGrid.interpolate( rVorton.position )
            stretchTilt = rVorton.vorticity * velJac # Usual way to compute stretching & tilting
            stretchTilt = numpy.asarray(stretchTilt)[0]
            rVorton.vorticity +=  0.5 * stretchTilt * timeStep # fudge factor for stability
            
    '''! \brief Diffuse vorticity using a particle strength exchange method.

    This routine partitions space into cells using the same grid
    as the "base vorton" grid.  Each vorton gets assigned to the
    cell that contains it.  Then, each vorton exchanges some
    of its vorticity with its neighbors in adjacent cells.

    This routine makes some simplifying assumptions to speed execution:

        -   Distance does not influence the amount of vorticity exchanged,
            except in as much as only vortons within a certain region of
            each other exchange vorticity.  This amounts to saying our kernel,
            eta, is a top-hat function.

        -   Theoretically, if an adjacent cell contains no vortons
            then this simulation should generate vorticity within
            that cell, e.g. by creating a new vorton in the adjacent cell.

        -   This simulation reduces the vorticity of each vorton, alleging
            that this vorticity is dissipated analogously to how energy
            dissipates at Kolmogorov microscales.  This treatment is not
            realistic but it retains qualitative characteristics that we
            want, e.g. that the flow dissipates at a rate related to viscosity.
            Dissipation in real flows is a more complicated phenomenon.

    \see Degond & Mas-Gallic (1989): The weighted particle method for
        convection-diffusion equations, part 1: the case of an isotropic viscosity.
        Math. Comput., v. 53, n. 188, pp. 485-507, October.

    \param timeStep - amount of time by which to advance simulation

    \param uFrame - frame counter

    \see StretchAndTiltVortons, AdvectVortons

    \note This routine assumes CreateInfluenceTree has already executed.

    '''
    def diffuseVorticityPSE(self, timeStep , uFrame ):

        ''' Phase 1: Partition vortons

        // Create a spatial partition for the vortons.
        // Each cell contains a dynamic array of integers
        // whose values are offsets into mVortons.'''
        #UniformGrid< Vector< unsigned > > ugVortRef( mInfluenceTree[0] ) ;
        ugVortRef = UniformGrid()
        ugVortRef.copyShape(self.influenceTree[0])
        ugVortRef.init([])

        numVortons = len(self.vortons)

        for vortonOffset in range(numVortons):
            # For each vorton...
            rVorton         = self.vortons[ vortonOffset ]
            # Insert the vorton's offset into the spatial partition.
            gridIndices = ugVortRef.indicesOfPosition( rVorton.position )
            vortonList = ugVortRef.getCell( gridIndices )
            vortonList.append(vortonOffset)
            ugVortRef.setCell(gridIndices, vortonList)

        # Phase 2: Exchange vorticity with nearest neighbors

        nx     = ugVortRef.getNumPoints( 0 )
        nxm1   = nx - 1
        ny     = ugVortRef.getNumPoints( 1 )
        nym1   = ny - 1
        nxy    = nx * ny
        nz     = ugVortRef.getNumPoints( 2 )
        nzm1   = nz - 1
        for idx_z in range(nzm1):
            #For all points along z except the last...
            offsetZ0 = idx_z         * nxy
            offsetZp = ( idx_z + 1 ) * nxy
            for idx_y in range(nym1):
                # For all points along y except the last...
                offsetY0Z0 =   idx_y       * nx + offsetZ0
                offsetYpZ0 = ( idx_y + 1 ) * nx + offsetZ0
                offsetY0Zp =   idx_y       * nx + offsetZp
                for idx_x in range(nxm1):
                    # For all points along x except the last...
                    offsetX0Y0Z0 = idx_x     + offsetY0Z0
                    vortonList = ugVortRef.getCell([idx_x,idx_y,idx_z])
                    vortonThereList = vortonList[:]
                    for rVortIdxHere in vortonList:
                        # For each vorton in this gridcell...
                        rVortonHere     = self.vortons[ rVortIdxHere ]
                        #rVorticityHere  = rVortonHere.vorticity
                        vortonThereList.remove(rVortIdxHere)
                        
                        # Diffuse vorticity with other vortons in this same cell:
                        for rVortIdxThere in vortonThereList:
                            # For each OTHER vorton within this same cell...
                            #const unsigned &    rVortIdxThere   = ugVortRef[ offsetX0Y0Z0 ][ ivThere ] ;
                            rVortonThere    = self.vortons[ rVortIdxThere ]
                            #rVorticityThere = rVortonThere.vorticity
                            vortDiff        = rVortonHere.vorticity - rVortonThere.vorticity
                            exchange        = 2.0 * self.viscosity * timeStep * vortDiff     # Amount of vorticity to exchange between particles.
                            rVortonHere.vorticity  -= exchange   # Make "here" vorticity a little closer to "there".
                            rVortonThere.vorticity += exchange   # Make "there" vorticity a little closer to "here".
                        
    
                        # Diffuse vorticity with vortons in adjacent cells:
                        #const unsigned offsetXpY0Z0 = idx[0] + 1 + offsetY0Z0 ; // offset of adjacent cell in +X direction
                        for rVortIdxThere in ugVortRef.getCell([ idx_x + 1, idx_y, idx_z ]):
                            #   // For each vorton in the adjacent cell in +X direction...
                            #    const unsigned &    rVortIdxThere   = ugVortRef[ offsetXpY0Z0 ][ ivThere ] ;
                            rVortonThere    = self.vortons[ rVortIdxThere ]
                            #rVorticityThere = rVortonThere.vorticity
                            vortDiff        = rVortonHere.vorticity - rVortonThere.vorticity
                            exchange        = self.viscosity * timeStep * vortDiff     # Amount of vorticity to exchange between particles.
                            rVortonHere.vorticity  -= exchange   # Make "here" vorticity a little closer to "there".
                            rVortonThere.vorticity += exchange   # Make "there" vorticity a little closer to "here".
                        
            
                        #const unsigned offsetX0YpZ0 = idx[0]     + offsetYpZ0 ; // offset of adjacent cell in +Y direction
                        for rVortIdxThere in ugVortRef.getCell([ idx_x, idx_y + 1, idx_z ]):
                            #   // For each vorton in the adjacent cell in +Y direction...
                            #    const unsigned &    rVortIdxThere   = ugVortRef[ offsetXpY0Z0 ][ ivThere ] ;
                            rVortonThere    = self.vortons[ rVortIdxThere ]
                            #rVorticityThere = rVortonThere.vorticity
                            vortDiff        = rVortonHere.vorticity - rVortonThere.vorticity
                            exchange        = self.viscosity * timeStep * vortDiff     # Amount of vorticity to exchange between particles.
                            rVortonHere.vorticity  -= exchange   # Make "here" vorticity a little closer to "there".
                            rVortonThere.vorticity += exchange   # Make "there" vorticity a little closer to "here".
                        
    
    
                        #const unsigned offsetX0Y0Zp = idx[0]     + offsetY0Zp ; // offset of adjacent cell in +Z direction
                        for rVortIdxThere in ugVortRef.getCell([ idx_x, idx_y, idx_z + 1 ]):
                            #   // For each vorton in the adjacent cell in +X direction...
                            #    const unsigned &    rVortIdxThere   = ugVortRef[ offsetXpY0Z0 ][ ivThere ] ;
                            rVortonThere    = self.vortons[ rVortIdxThere ]
                            #rVorticityThere = rVortonThere.vorticity
                            vortDiff        = rVortonHere.vorticity - rVortonThere.vorticity
                            exchange        = self.viscosity * timeStep * vortDiff     # Amount of vorticity to exchange between particles.
                            rVortonHere.vorticity  -= exchange   # Make "here" vorticity a little closer to "there".
                            rVortonThere.vorticity += exchange   # Make "there" vorticity a little closer to "here".
                        
    
                        # Dissipate vorticity.  See notes in header comment.
                        rVortonHere.vorticity  -= self.viscosity * timeStep * rVortonHere.vorticity   # Reduce "here" vorticity.

    '''! \brief Advect vortons using velocity field

    \param timeStep - amount of time by which to advance simulation

    \see ComputeVelocityGrid

    '''
    def advectVortons(self, timeStep ):
        #f = open("/tmp/vorton.dmp", "w")
        #f1 = open("/tmp/vorton2.dmp", "w")
        for rVorton in self.vortons:
            # For each vorton...
            #Vorton & rVorton = mVortons[ offset ] ;
            #Vec3 velocity ;
            #f.write(str(rVorton.position))
            velocity = self.velGrid.interpolate( rVorton.position )
            rVorton.position += velocity * timeStep
            rVorton.velocity = velocity # Cache this for use in collisions with rigid bodies.
            #f1.write(str(rVorton.position))
        #f.close()
        #f1.close()


    '''! \brief Advect (subset of) passive tracers using velocity field

    \param timeStep - amount of time by which to advance simulation

    \param uFrame - frame counter

    \param itStart - index of first tracer to advect

    \param itEnd - index of last tracer to advect

    \see AdvectTracers
    '''
    def advectTracers(self, timeStep , uFrame):
        #f = open("sim.out/tracer"+ str(uFrame)+".obj", "w")
        #largestVel = 0.0
        #print f
        print("copy grid cont")
        f2 = open("sim.out/grid","w")
        
        
        zi = 0
        for gz in self.velGrid.contents:
            yi = 0
            for gy in gz:
                xi = 0
                for gx in gy:
                    f2.write(str([zi,yi,xi]) + " ")
                    f2.write(str(gx) + "\n")
                    xi += 1
                yi+=1
            zi+=1
        f2.close()
        #grid.grid.setgridcontents(self.velGrid.contents,len(self.velGrid.contents),len(self.velGrid.contents[0]),len(self.velGrid.contents[0][0]))
        cellsPerExtent = self.velGrid.getCellsPerExtent()
        #grid.grid.setcellsperextent(cellsPerExtent)
        '''tracerPositions = [[],[],[]]
        for rTracer in self.tracers:
            tracerPositions[0].append(rTracer.position)
            idx = self.velGrid.indicesOfPosition( rTracer.position )
            tracerPositions[1].append(idx)
            vMinCorner = self.velGrid.positionFromIndices( idx )
            tracerPositions[2].append(vMinCorner)
            
        print len(self.velGrid.contents),len(self.velGrid.contents[0]),len(self.velGrid.contents[0][0])
        print "inter"
        
        tracerPositions = grid.grid.advecttracers(tracerPositions)
        for rTracer,pos in zip(self.tracers,tracerPositions):
            rTracer.position = pos'''
        #if len(self.velGrid.contents) == 17:
        #    print "stop"
        for rTracer in self.tracers:
            # For each passive tracer in this slice...
            #Particle & rTracer = mTracers[ offset ] ;
            #Vec3 velocity ;
            #f.write("v " + str(rTracer.position[0]) + " " + str(rTracer.position[1]) + " " + str(rTracer.position[2]) + "\n")
            
            velocity = self.velGrid.interpolate(rTracer.position )
            '''indices = self.velGrid.indicesOfPosition( rTracer.position )
            findices = [0,0,0]
            findices[0] = indices[0] + 1
            findices[1] = indices[1] + 1
            findices[2] = indices[2] + 1
            vMinCorner = self.velGrid.positionFromIndices( indices )'''
            
            #velocity = grid.grid.interpolate( rTracer.position, findices,  vMinCorner, cellsPerExtent)
            
            #print "v"
            #print grid.grid.getcellcontents(findices)
            #print self.velGrid.getCell(indices)
            #print velocity
            #print vel
            rTracer.position += velocity * timeStep
            rTracer.velocity  = velocity # Cache for use in collisions
            #velMag = velocity[0]**2 + velocity[1]**2 + velocity[2]**2
            #if  velMag > largestVel:
            #largestVel = velMag
            #print velocity
                
        #rib.write(uFrame,self.tracers)
        print("inter finished")
        #f.write("f 1 2 3 4")
        #f.close()
        Output.vtk.writeGrid("sim.out/grid" + str(uFrame),self.velGrid)
        Output.vtk.write( "sim.out/tracer" + str(uFrame), self.tracers )
    ''' \brief Advect passive tracers using velocity field

    \param timeStep - amount of time by which to advance simulationc

    \param uFrame - frame counter

    \see AdvectVortons

    '''
    '''def advectTracers(self, timeStep ,  uFrame ):

        numTracers = len(self.tracers)

        #if USE_TBB
        // Estimate grain size based on size of problem and number of processors.
        const size_t grainSize =  MAX2( 1 , numTracers / gNumberOfProcessors ) ;
        // Advect tracers using multiple threads.
        parallel_for( tbb::blocked_range<size_t>( 0 , numTracers , grainSize ) , VortonSim_AdvectTracers_TBB( this , timeStep , uFrame ) ) ;
        #else'
        self.advectTracersSlice( timeStep , uFrame , 0 , numTracers )
        #endif'''
        
                                
    '''! \brief Update vortex particle fluid simulation to next time.

    \param timeStep - incremental amount of time to step forward

    \param uFrame - frame counter, used to generate files

    '''
    def update(self, timeStep , uFrame ):

        #QUERY_PERFORMANCE_ENTER ;
        #print "creatInfluenceTree"
        self.createInfluenceTree()
        #QUERY_PERFORMANCE_EXIT( VortonSim_CreateInfluenceTree ) ;

        #QUERY_PERFORMANCE_ENTER ;
        print("ComputeVelGrid")
        self.computeVelocityGrid()
        #QUERY_PERFORMANCE_EXIT( VortonSim_ComputeVelocityGrid ) ;
        #self.dumpVelGrid()
        #QUERY_PERFORMANCE_ENTER ;
        print("stretchAndTiltVortons")
        self.stretchAndTiltVortons( timeStep , uFrame )
        #QUERY_PERFORMANCE_EXIT( VortonSim_StretchAndTiltVortons ) ;
        #self.dumpVelGrid()
        #QUERY_PERFORMANCE_ENTER ;
        #print "diffuseVorticity"
        self.diffuseVorticityPSE( timeStep , uFrame )
        #QUERY_PERFORMANCE_EXIT( VortonSim_DiffuseVorticityPSE ) ;

        #QUERY_PERFORMANCE_ENTER ;
        #print "advect vortons"
        self.advectVortons( timeStep )
        #QUERY_PERFORMANCE_EXIT( VortonSim_AdvectVortons ) ;

        #QUERY_PERFORMANCE_ENTER ;
        #print "advect tracers"
        #print timeStep, uFrame
        self.advectTracers( timeStep , uFrame )
        #QUERY_PERFORMANCE_EXIT( VortonSim_AdvectTracers ) ;
