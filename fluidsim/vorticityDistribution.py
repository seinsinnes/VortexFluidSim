'''
Created on 12/01/2012

@author: christopherd
'''

'''! \brief Specify vorticity in the shape of a vortex ring.

    The vorticity specified by this class derives from taking the curl of
    a localized jet.  The vorticity is therefore guaranteed to be solenoidal,
    to within the accuracy the discretization affords.

    \see VortexRing

'''
import sys
import math
from numpy import array
from numpy.random import *
import numpy
from Space import uniformGrid
from fluidsim.vorton import Vorton
import random

class VortexRing:

        '''! \brief Initialize parameters for a vortex ring

            The vorticity profile resulting from this is such that the vorticity is in [0,1].

            \param fRadius - radius of vortex ring core

            \param fThickness - thickness of vortex ring, i.e. radius of annular core

            \param vDirection - vector of ring axis, also vector of propagation
        '''

        def __init__(self, fRadius , fThickness ,  vDirection ):
            self.radius = fRadius
            self.thickness = fThickness
            self.direction = vDirection

        def getDomainSize(self):
            boxSideLength   = 2. * ( self.radius + self.thickness)   # length of side of virtual cube
            return array([ 1.0 , 1.0 , 1.0 ]) * boxSideLength

        def assignVorticity(self, position , vCenter ):
            vFromCenter     = position - vCenter              # displacement from ring center to vorton position
            tween = numpy.dot(vFromCenter, self.direction)    # projection of position onto axis
            vPtOnLine = vCenter + self.direction * tween  # closest point on axis to vorton position
            vRho = position - vPtOnLine  # direction radially outward from annulus core
            rho = numpy.sqrt(numpy.vdot(vRho, vRho))
            distAlongDir    = numpy.dot(self.direction, vFromCenter)        # distance along axis of vorton position
            radCore         = numpy.sqrt( (rho - self.radius )**2 + distAlongDir**2 ) ; # distance from annular core

            if( radCore < self.thickness ):
                vortProfile     = 0.5 * ( numpy.cos( numpy.pi * radCore / self.thickness ) + 1.0 )
                vortPhi         = vortProfile
                rhoHat          = vRho                       # direction radially away from annular core
                rhoHat.normalize()
                phiHat = numpy.cross(self.direction, rhoHat)         # direction along annular core
                vorticity                       = vortPhi * phiHat
            else:
                vorticity = array([ 0.0 , 0.0 , 0.0 ])

            return vorticity


class JetRing:
        '''! \brief Initialize parameters for a vortex ring (using a different formula from the other).

            The vorticity profile resulting from this is such that the induced velocity is in [0,1].

            \param fRadiusSlug - radius of central region where velocity is constant

            \param fThickness - thickness of vortex ring, i.e. radius of annular core

            \param vDirection - vector of ring axis, also vector of propagation

            \param fSpeed   - speed of slug

        '''
        def __init__(self, fRadiusSlug , fThickness , vDirection ):
            self.radiusSlug = fRadiusSlug
            self.thickness = fThickness
            self.radiusOuter = fRadiusSlug + self.thickness
            self.direction = vDirection
        

        def getDomainSize(self):
            boxSideLength   = 2.0 * self.radiusOuter  # length of side of virtual cube
            return array([ 1.0 , 1.0 , 1.0 ]) * boxSideLength

        def assignVorticity(self, position , vCenter ):
            vFromCenter     = position - vCenter              # displacement from ring center to vorton position
            tween           = numpy.dot(vFromCenter, self.direction)      # projection of position onto axis
            vPtOnLine       = vCenter + self.direction * tween    # closest point on axis to vorton position
            vRho            = position - vPtOnLine            # direction radially outward from annulus core
            rho             = numpy.sqrt(numpy.vdot(vRho,vRho))                # Get the magnitude of VRho.. distance from axis
            distAlongDir    = numpy.dot(self.direction, vFromCenter)        # distance along axis of vorton position
            rhoHat = vRho / rho                                             #Normalise vRho as unit vecotr
            if rho < self.radiusOuter and rho > self.radiusSlug:
                #Probe position is inside jet region.
                if  abs( distAlongDir ) < self.radiusSlug:
                    streamwiseProfile = 0.5 * ( numpy.cos( numpy.pi * distAlongDir / self.radiusSlug ) + 1.0 )
                else:
                    streamwiseProfile   =  0.0
                    
                radialProfile       = numpy.sin( numpy.pi * ( rho - self.radiusSlug ) / self.thickness )
                vortPhi             = streamwiseProfile * radialProfile * numpy.pi / self.thickness
                phiHat = numpy.cross(self.direction, rhoHat)      # direction along annular core
                vorticity                           = vortPhi * phiHat
            else:
                vorticity = array([ 0.0 , 0.0 , 0.0 ])
                
            return vorticity


''' \brief Create a vorticity field using vortex particles

    \param vortons - (out) array of vortex particles

    \param fMagnitude - maximum value of vorticity in the ring

    \param vortexRing - characteristics of a vortex ring

    \param numVortonsMax - maximum number of vortons this routine will generate.
        This routine may (and likely will) generate fewer vortons than this.
        This effectivly specifies the density of vortons, i.e. how finely resolved the vortons will be.
        Suggested value provided should be at least 512, which corresponds to an 8x8x8 grid.
'''

def assignVorticity( vortons , fMagnitude , numVortonsMax , vorticityDistribution, vCenter ):

    vDimensions     = vorticityDistribution.getDomainSize() # length of each side of grid box
    #vCenter  =       array([ 0.0 , 0.0 , 0.0 ])                        # Center of vorticity distribution
    vMin     =       vCenter - 0.5 * vDimensions              # Minimum corner of box containing vortons
    vMax     =       vMin + vDimensions                       # Maximum corner of box containing vortons
    skeleton =      uniformGrid.UniformGridGeometry( numVortonsMax , vMin , vMax , True )
    
    numCells     = [ max( 1 , skeleton.getNumCells(0)), \
                       max( 1 , skeleton.getNumCells(1)), \
                       max( 1 , skeleton.getNumCells(2)) ]  # number of grid cells in each direction of virtual uniform grid
    
    # Total number of cells should be as close to numVortonsMax as possible without going over.
    # Worst case allowable difference would be numVortonsMax=7 and numCells in each direction is 1 which yields a ratio of 1/7.
    # But in typical situations, the user would like expect total number of virtual cells to be closer to numVortonsMax than that.
    # E.g. if numVortonsMax=8^3=512 somehow yielded numCells[0]=numCells[1]=numCells[2]=7 then the ratio would be 343/512~=0.67.
    
    print numCells
    print vMin
    print vMax
    print numVortonsMax
    f = open("/tmp/vortons.dmp", "w")
    while numCells[0] * numCells[1] * numCells[2] > numVortonsMax:
        # Number of cells is excessive.
        # This can happen when the trial number of cells in any direction is less than 1 -- then the other two will likely be too large.
        numCells[0] = max( 1 , int(numCells[0] / 2) )
        numCells[1] = max( 1 , int(numCells[1] / 2) )
        numCells[2] = max( 1 , int(numCells[2] / 2) )
    

    oneOverN     = [ 1.0 / float( numCells[0] ) , 1.0 / float( numCells[1] ) , 1.0 / float( numCells[2] ) ]
    gridCellSize  = array([ vDimensions[0] * oneOverN[0] , vDimensions[1] * oneOverN[1] , vDimensions[2] * oneOverN[2] ])
    vortonRadius    = pow( gridCellSize[0] * gridCellSize[1] * gridCellSize[2] , 1.0 / 3.0 ) * 0.5
    if  0.0 == vDimensions[2]:
        #z size is zero, so domain is 2D.
        vortonRadius = pow( gridCellSize[0] * gridCellSize[1] , 0.5 ) * 0.5
    
    vNoise  = 0.0 * gridCellSize

    
    index = [0,0,0]  # index of each position visited
    # Iterate through each point in a uniform grid.
    # If probe position is inside vortex core, add a vorton there.
    # This loop could be rewritten such that it only visits points inside the core,
    # but this loop structure can readily be reused for a wide variety of configurations.
    for index[2] in range(0,numCells[2]):
        # For each z-coordinate...
        position_z = ( float( index[2] ) + 0.25 ) * gridCellSize[2] + vMin[2]
        for index[1] in range( 0, numCells[1] ):
            # For each y-coordinate...
            position_y = ( float( index[1] ) + 0.25 ) * gridCellSize[1] + vMin[1]
            for index[0] in range( 0, numCells[0] ):
                # For each x-coordinate...
                position_x = ( float( index[0] ) + 0.25 ) * gridCellSize[0] + vMin[0]
                position = array([ position_x , position_y , position_z ])                              # vorton position
                position += vNoise * ( random.random() - 0.5 ) 
                vorticity = vorticityDistribution.assignVorticity( position , vCenter )
                vorton = Vorton( position, vorticity * fMagnitude , vortonRadius )
                floatEpsilon = sys.float_info.epsilon
                floatMin = sys.float_info.min
                sTiny = math.exp( 0.5 * ( math.log( floatEpsilon ) + math.log( floatMin ) ) )
                
                if numpy.sqrt(numpy.vdot(vorticity,vorticity)) > sTiny:
                    #Vorticity is significantly non-zero.
                    print vorton.radius
                    f.write(str(vorton.position) + str(vorton.vorticity) +"\n")
                    vortons.append( vorton )
  
    f.close()