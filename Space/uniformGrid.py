'''
Created on 13/01/2012

@author: christopherd
'''

import sys
from numpy import array
import numpy
import copy

def nearestPowerOfTwo(val):
    pot = 1
    while val != 0:
        val = int(val/2)
        pot = pot*2
    
    return pot

''' \brief Base class for uniform grid.

    The shape of this grid is such that the "minimal corner" point
    resides at indices {0,0,0} and the "maximal corner" point
    resides at indices {Nx-1,Ny-1,Nz-1}.

    The number of /points/ in each direction i is N_i.

    A cell is defined by the 8 points that lie at its corners.

    This also implies that the grid must have at least 2 points in
    each direction.

    The number of /cells/ in each direction i is (N_i-1).

    The size of a side i of each cell is therefore
        s_i = (vMax-vMin)_i / (N_i-1) .


                  vMin
                   0       1      ...     Nx-1
                  (*)------*-------*-------* 0
                 ./|       |       |       |
                ./ |       |       |       |
               .*  |       |       |       |
               /|  *-------*-------*-------* 1
              / | /|       |       |       |
        Nz-1 *  |/ |       | cell  |       |
             |  *  |       |       |       | .
             | /|  *-------*-------*-------* .
             |/ | /|       |       |       | .
             *  |/ |       |       |       |
             |  *  |       |       |       |
             | /|  *-------*-------*-------* Ny-1
             |/ | /       /       /       /
             *  |/       /       /       /
             |  *-------*-------*-------*
             | /       /       /       /
             |/       /       /       /
             *-------*-------*------(*)vMax


'''
class UniformGridGeometry:


        '''! \brief Construct a uniform grid that fits the given geometry.

            \see Clear, DefineShape
        '''
        def __init__(self, uNumElements , vMin , vMax ,bPowerOf2 ):
            if uNumElements:
                self.defineShape( uNumElements , vMin , vMax , bPowerOf2 )
            else:
                self.numPoints = [0.0,0.0,0.0]
   
            
        ''' \brief Define the shape a uniform grid such that it fits the given geometry.

            \param uNumElements - number of elements this container will contain.

            \param vMin - minimal coordinate of axis-aligned bounding box.

            \param vMax - maximal coordinate of axis-aligned bounding box.

            \param bPowerOf2 - whether to make each grid dimension a power of 2.
                Doing so simplifies grid subdivision, if this grid will be used in a hierarchical grid.

            This makes a uniform grid of cells, where each cell is the same size
            and the side of each cell is nearly the same size.  If the cells are
            3-dimensional then that means each cell is a box, nearly a cube.
            The number of dimensions of the region depends on the actual size of
            the region.  If any size component is smaller than a small threshold
            then this class considers that component to be zero, and reduces the
            dimensionality of the region.  For example, if the region size is
            (2,3,0) then this class considers the region to have 2 dimensions
            (x and y) since the z size is zero.  In this example, the cells
            would be nearly square rectangles (instead of boxes).

        '''
        def defineShape(self, uNumElements , vMin , vMax , bPowerOf2 ):
            
            self.minCorner  = vMin
            floatEpsilon = sys.float_info.epsilon
            nudge = 1.0 + floatEpsilon  #slightly expand size to ensure robust containment even with roundoff
            self.gridExtent     = ( vMax - vMin ) * nudge
            #print "nume:"
            #print uNumElements
            #print vMax
            #print vMin
            #print bPowerOf2
            vSizeEffective = self.getExtent()
            #print vSizeEffective
            numDims = 3   # Number of dimensions to region.
            if 0.0 == vSizeEffective[0]:
                #X size is zero so reduce dimensionality
                vSizeEffective[0] = 1.0 # This component will not contribute to the total region volume/area/length.
                self.gridExtent[0] = 0.0
                numDims -= 1
                
            if 0.0 == vSizeEffective[1]:
                #Y size is zero so reduce dimensionality
                vSizeEffective[1] = 1.0 # This component will not contribute to the total region volume/area/length.
                self.gridExtent[1] = 0.0
                numDims -= 1
    
            if 0.0 == vSizeEffective[2]:
                #Z size is zero so reduce dimensionality
                vSizeEffective[2] = 1.0 #This component will not contribute to the total region volume/area/length.
                self.gridExtent[2] = 0.0
                numDims -= 1
            

            # Compute region volume, area or length (depending on dimensionality).
            volume              = vSizeEffective[0] * vSizeEffective[1] * vSizeEffective[2]
            cellVolumeCubeRoot  = pow( volume / float( uNumElements ) , -1.0 / float( numDims ) ) # Approximate size of each cell in grid.
            # Compute number of cells in each direction of uniform grid.
            # Choose grid dimensions to fit as well as possible, so that the total number
            # of grid cells is nearly the total number of elements in the contents.
            #print self.getExtent()[0]
            #print cellVolumeCubeRoot
            numCells = [ max( 1 , int( self.getExtent()[0] * cellVolumeCubeRoot + 0.5 ) ) ,
                                     max( 1 , int( self.getExtent()[1] * cellVolumeCubeRoot + 0.5 ) ) ,
                                     max( 1 , int( self.getExtent()[2] * cellVolumeCubeRoot + 0.5 ) ) ]
            #print numCells
            if bPowerOf2:
                # Choose number of gridcells to be powers of 2.
                # This will simplify subdivision in a NestedGrid.
                numCells[ 0 ] = nearestPowerOfTwo( numCells[ 0 ] )
                numCells[ 1 ] = nearestPowerOfTwo( numCells[ 1 ] )
                numCells[ 2 ] = nearestPowerOfTwo( numCells[ 2 ] )
            
            #print numCells
            while numCells[ 0 ] * numCells[ 1 ] * numCells[ 2 ] >= uNumElements * 8:
                # Grid capacity is excessive.
                # This can occur when the trial numCells is below 0.5 in which case the integer arithmetic loses the subtlety.
                numCells[ 0 ] = max( 1 , numCells[0] / 2 )
                numCells[ 1 ] = max( 1 , numCells[1] / 2 )
                numCells[ 2 ] = max( 1 , numCells[2] / 2 )
            
            self.numPoints = [ 0,0,0 ]
            self.numPoints[ 0 ] = numCells[ 0 ] + 1 # Increment to obtain number of points.
            self.numPoints[ 1 ] = numCells[ 1 ] + 1 # Increment to obtain number of points.
            self.numPoints[ 2 ] = numCells[ 2 ] + 1 # Increment to obtain number of points.

            #print "numpTs:"
            #print self.numPoints
            self.precomputeSpacing()
            
            
        '''! \brief Precompute grid spacing, to optimize OffsetOfPosition and other utility routines.
        '''
        def precomputeSpacing(self):
            self.cellExtent = array([0.0,0.0,0.0])
            self.cellExtent[0]       = float(self.getExtent()[0]) / float( self.getNumCells( 0 ) )
            self.cellExtent[1]       = float(self.getExtent()[1]) / float( self.getNumCells( 1 ) )
            self.cellExtent[2]       = float(self.getExtent()[2]) / float( self.getNumCells( 2 ) )
            #print "cellextent", self.cellExtent
            
            self.cellsPerExtent = array([0.0,0.0,0.0])
            self.cellsPerExtent[0]   = float( self.getNumCells( 0 ) ) / float(self.getExtent()[0])
            self.cellsPerExtent[1]   = float( self.getNumCells( 1 ) ) / float(self.getExtent()[1])
            
            print("c e")
            print(self.getNumCells( 1 ))
            print(self.getExtent())
            
            if 0.0 == self.getExtent()[2]:
                # Avoid divide-by-zero for 2D domains that lie in the XY plane.
                floatMin = sys.float_info.min
                self.cellsPerExtent[2]   = 1.0 / floatMin
            else:
                self.cellsPerExtent[2]   = float( self.getNumCells( 2 ) ) / float(self.getExtent()[2])
        
        
        '''! \brief Get world-space dimensions of UniformGridGeometry
        '''
        def getExtent(self):
            return self.gridExtent
        
        ''' \brief Get reciprocal of cell spacing.
        '''
        def getCellsPerExtent( self):
            return self.cellsPerExtent

        def getNumCells(self, index ):
            return self.getNumPoints( index ) - 1
        
        def getNumPoints(self, index ):
            return int(self.numPoints[ index ])
        
        def getMinCorner( self ):
            return self.minCorner
        
        def getGridCapacity(self):
            return self.getNumPoints(0) * self.getNumPoints(1) * self.getNumPoints(2)
        
        ''' \brief Create a lower-resolution uniform grid based on another

            \param src - Source uniform grid upon which to base dimensions of this one

            \param iDecimation - amount by which to reduce the number of grid cells in each dimension.
                Typically this would be 2.

            \note The number of cells is decimated.  The number of points is different.

        '''
        def decimate(self, src , iDecimation ):
            self.gridExtent         = src.gridExtent
            self.minCorner          = src.minCorner
            self.numPoints[ 0 ]     = src.getNumCells( 0 ) / iDecimation + 1
            self.numPoints[ 1 ]     = src.getNumCells( 1 ) / iDecimation + 1
            self.numPoints[ 2 ]     = src.getNumCells( 2 ) / iDecimation + 1
            if( iDecimation > 1 ):
                # Decimation could reduce dimension and integer arithmetic could make value be 0, which is useless if src contained any data.
                self.numPoints[ 0 ] = max( 2, self.getNumPoints( 0 ) )
                self.numPoints[ 1 ] = max( 2, self.getNumPoints( 1 ) )
                self.numPoints[ 2 ] = max( 2, self.getNumPoints( 2 ) )

            #print self.numPoints
            self.precomputeSpacing()
            
        
        ''' \brief Compute indices into contents array of a point at a given position

            \param vPosition - position of a point.  It must be within the region of this container.

            \param indices - Indices into contents array of a point at vPosition.

            \see IndicesFromOffset, PositionFromOffset, OffsetOfPosition.

            \note Derived class defines the actual contents array.

        '''
        def indicesOfPosition(self, vPosition ):
            # Notice the pecular test here.  vPosition may lie slightly outside of the extent give by vMax.
            # Review the geometry described in the class header comment.
            vPosRel = vPosition - self.getMinCorner() # position of given point relative to container region
            vIdx = [ int(vPosRel[0] * self.getCellsPerExtent()[0]) , int(vPosRel[1] * self.getCellsPerExtent()[1]) , int(vPosRel[2] * self.getCellsPerExtent()[2]) ] 
            #print self.getCellsPerExtent()
            return vIdx
    
        ''' \brief Compute position of minimal corner of grid cell with given indices

            \param position - position of minimal corner of grid cell

            \param indices - grid cell indices.

            \note Rarely if ever would you want to compute position from indices in this way.
                    Typically, this kind of computation occurs inside a triply-nested loop,
                    in which case the procedure should compute each component
                    separately.  Furthermore, such a routine would cache
                    GetCellSpacing instead of computing it each iteration.

        '''
        def positionFromIndices( self , indices ):
            vPosition = array([0.0,0.0,0.0], dtype=numpy.float32)
            #print self.getCellSpacing()
            vPosition[0] = self.getMinCorner()[0] + float( indices[0] ) * self.getCellSpacing()[0]
            vPosition[1] = self.getMinCorner()[1] + float( indices[1] ) * self.getCellSpacing()[1]
            vPosition[2] = self.getMinCorner()[2] + float( indices[2] ) * self.getCellSpacing()[2]
            return vPosition
        
        ''' \brief Return extent (in world units) of a grid cell.
        '''
        def getCellSpacing( self):
            return self.cellExtent
         
        ''' \brief Copy shape information from another UniformGrid into this one
        '''
        def copyShape(self, src ):
            self.decimate( src , 1 )
         
        
''' \brief Templated container for fast spatial lookups and insertions
'''
class UniformGrid(UniformGridGeometry):


        ''' \brief Construct a uniform grid container that fits the given geometry.
            \see Initialize
        '''
        def __init__(self, uNumElements = None, vMin = None , vMax = None, bPowerOf2 = False):
            UniformGridGeometry.__init__(self, uNumElements, vMin, vMax, bPowerOf2)
            self.interpolateCombinations = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]]
            interpolateCombinationsReverse = self.interpolateCombinations[:]
            interpolateCombinationsReverse.reverse()
            self.interpolateCombinations = zip(self.interpolateCombinations, interpolateCombinationsReverse)
            
            self.init()
            
            
        def init(self, cellTemplate = sys.float_info.min):
            #print "number_of_points"
            numPoints_z = self.getNumPoints( 2 )
            numPoints_y = self.getNumPoints( 1 )
            numPoints_x = self.getNumPoints( 0 )
            
            self.contents = []
            yzplane = []
            zfibre = []
            for idx_x in range(numPoints_x):
                for idx_y in range(numPoints_y):
                    for idx_z in range(numPoints_z):
                        zfibre.append(copy.deepcopy(cellTemplate))
                    yzplane.append(zfibre)
                    zfibre = []
                self.contents.append(yzplane)
                yzplane = []
                        
            #print self.contents

        def getCell(self, indices):
            #            print "in get cell"
            #            print len(self.contents)
            #            print indices
            
            return self.contents[int(indices[0])][int(indices[1])][int(indices[2])]
        
        def setCell(self, indices, value):
            self.contents[int(indices[0])][int(indices[1])][int(indices[2])] = value

        def dumpCells(self):
            dump = ""
            numPoints_z = self.getNumPoints( 2 )
            numPoints_y = self.getNumPoints( 1 )
            numPoints_x = self.getNumPoints( 0 )
            for idx_x in range(numPoints_x):
                for idx_y in range(numPoints_y):
                    for idx_z in range(numPoints_z):
                        dump += self.contents[idx_x][idx_y][idx_z].dumpSelf()
                        
            return dump
        
        ''' \brief Interpolate values from grid to get value at given position

            \param vPosition - position to sample

            \return Interpolated value corresponding to value of grid contents at vPosition.

        '''
        def interpolate(self, vPosition ):
            #unsigned        indices[3] ; // Indices of grid cell containing position.
            indices = self.indicesOfPosition( vPosition )
            #print "i", indices
            #print vPosition
            #Vec3            vMinCorner ;
            vMinCorner = self.positionFromIndices( indices )
            #const unsigned  offsetX0Y0Z0 = OffsetFromIndices( indices ) ;
            vDiff         = vPosition - vMinCorner # Relative location of position within its containing grid cell.
            tween         = array([ vDiff[0] * self.getCellsPerExtent()[0] , vDiff[1] * self.getCellsPerExtent()[1] , vDiff[2] * self.getCellsPerExtent()[2]])
            oneMinusTween = array([ 1.0 , 1.0 , 1.0] ) - tween
            #const unsigned  numXY         = GetNumPoints( 0 ) * GetNumPoints( 1 ) ;
            #const unsigned  offsetX1Y0Z0  = offsetX0Y0Z0 + 1 ;
            x0y0z0 = array(indices)
            x1y0z0 = x0y0z0 + [1,0,0]
            #const unsigned  offsetX0Y1Z0  = offsetX0Y0Z0 + GetNumPoints(0) ;
            x0y1z0 = x0y0z0 + [0,1,0]
            #const unsigned  offsetX1Y1Z0  = offsetX0Y0Z0 + GetNumPoints(0) + 1 ;
            x1y1z0 = x0y0z0 + [1,1,0]
            #const unsigned  offsetX0Y0Z1  = offsetX0Y0Z0 + numXY ;
            x0y0z1 = x0y0z0 + [0,0,1]
            
            #const unsigned  offsetX1Y0Z1  = offsetX0Y0Z0 + numXY + 1 ;
            x1y0z1 = x0y0z0 + [1,0,1]
            #const unsigned  offsetX0Y1Z1  = offsetX0Y0Z0 + numXY + GetNumPoints(0) ;
            x0y1z1 = x0y0z0 + [0,1,1]
            #const unsigned  offsetX1Y1Z1  = offsetX0Y0Z0 + numXY + GetNumPoints(0) + 1 ;
            x1y1z1 = x0y0z0 + [1,1,1]
            
            #vResult = self.getCell(indices)
            #for c,rc in self.interpolateCombinations:
                #cellVal = self.getCell([indices[0] + c[0], indices[1] + c[1], indices[2] + c[2]])
                #aTween = (rc[0] + (-rc[0])*tween[0]) * (rc[1] + (-rc[1])*tween[1]) * (rc[2] + (-rc[2])*tween[2])
                #vResult += aTween*cellVal
            #print "v c"
            #print vDiff
            #print self.cellExtent  
            #print tween
            vResult = oneMinusTween[0] * oneMinusTween[1] * oneMinusTween[2] * self.getCell(x0y0z0) \
            +         tween[0] * oneMinusTween[1] * oneMinusTween[2] * self.getCell(x1y0z0 ) \
            + oneMinusTween[0] *         tween[1] * oneMinusTween[2] * self.getCell(x0y1z0 ) \
            +         tween[0] *         tween[1] * oneMinusTween[2] * self.getCell(x1y1z0 ) \
            + oneMinusTween[0] * oneMinusTween[1] *         tween[2] * self.getCell(x0y0z1 ) \
            +         tween[0] * oneMinusTween[1] *         tween[2] * self.getCell(x1y0z1 ) \
            + oneMinusTween[0] *         tween[1] *         tween[2] * self.getCell(x0y1z1 ) \
            +         tween[0] *         tween[1] *         tween[2] * self.getCell(x1y1z1 )
            
            return vResult
