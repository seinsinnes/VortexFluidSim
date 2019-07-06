from Space.uniformGrid import UniformGrid

class NestedGrid:

        ''' \brief Construct a blank nested uniform grid spatial partition
        '''
        def __init__(self):
            self.decimations = []
            self.layers = []
        
        
        ''' \brief Initialize an unpopulated nested uniform grid spatial partition, based on a given UniformGrid

            \param src - UniformGrid upon which this NestedGrid is based.
        '''
            
        def initialise(self, src):
            self.layers = []
            numLayers = self.precomputeNumLayers( src )
            #mLayers.Reserve( numLayers ) # Preallocate number of layers to avoid reallocation during PushBack.
            self.addLayer( src , 1 )
            index = 1
            while self.layers[ index-1 ].getGridCapacity() > 8: # /* a cell has 8 corners */
                # Layer to decimate has more than 1 cell.
                self.addLayer( self.layers[ index - 1 ] , 2 ) # Initialize child layer based on decimation of its parent grid.
                index += 1

            self.precomputeDecimations()
        
        '''! \brief Precompute the total number of layers this nested grid will contain

            \param src - UniformGrid upon which this NestedGrid is based.

        '''
        def precomputeNumLayers(self, src ):
            numLayers      = 1    # Tally src layer.
            numPoints   = [ src.getNumPoints( 0 ) , src.getNumPoints( 1 ) , src.getNumPoints( 2 ) ]
            size = numPoints[0] * numPoints[1] * numPoints[2]
            while( size > 8  ): #/* a cell has 8 corners */
                # Layer has more than 1 cell.
                numLayers += 1
                # Decimate number of cells (where #cells = #points-1):
                numPoints[0] = max( ( numPoints[0] - 1 ) / 2 , 1 ) + 1
                numPoints[1] = max( ( numPoints[1] - 1 ) / 2 , 1 ) + 1
                numPoints[2] = max( ( numPoints[2] - 1 ) / 2 , 1 ) + 1
                size = numPoints[0] * numPoints[1] * numPoints[2]
                
            return numLayers
        
        ''' \brief Add a layer to the top of the nested grid.

            \param layerTemplate - UniformGridGeometry defining child layer.

            \param iDecimation - amount by which to decimate child layer, in each direction.

            This facilitates building the tree from leaves to root.
            This method also pre-allocates memory for the newly added layer,
            and initializes its contents to whatever the default constructor returns.

        '''
        def addLayer(self, layerTemplate , iDecimation ):
            self.layers.append(UniformGrid())
            self.layers[-1].decimate( layerTemplate , iDecimation )
            self.layers[-1].init(layerTemplate.getCell([0,0,0]))
            
        def getDepth(self):
            return len(self.layers)
        
        def __getitem__(self, key):
            return self.layers[key]
        
        '''! \brief Compute decimations, in each direction, for specified parent layer

            \param decimations - (out) ratio of dimensions between child layer and its parent.

            \param iParentLayer - index of parent layer.
                                Child has index iParentLayer-1.
                                Layer 0 has no child so providing "0" is invalid.

            This method effectively gives the number of child cells in each
            grid cluster that a parent cell represents.

            Each non-leaf layer in this NestedGrid is a decimation of its child
            layer. Typically that decimation is 2 in each direction, but the
            decimation can also be 1, or, more atypically, any other integer.
            Each child typically has twice as many cells in each direction as
            its parent.

            \note This assumes each parent has an integer decimation of its child.

            \see GetDecimations

        '''
        def computeDecimations(self, decimations , iParentLayer ):
            parent = self[ iParentLayer     ]
            child  = self[ iParentLayer - 1 ]
            decimations[ 0 ] = child.getNumCells( 0 ) / parent.getNumCells( 0 )
            decimations[ 1 ] = child.getNumCells( 1 ) / parent.getNumCells( 1 )
            decimations[ 2 ] = child.getNumCells( 2 ) / parent.getNumCells( 2 )
            print(decimations)
        
        '''! \brief Precompute decimations for each layer.

            This provides the number of grid cells per cluster
            of a child of each layer.

            \note The child layer has index one less than the parent layer index.
                    That implies there is no such thing as "parent layer 0".
                    Layer 0  has no children. That further implies there is no
                    meaningful value for decimations at iParentLayer==0.

        '''
        def precomputeDecimations(self ):

            numLayers = self.getDepth() 

            for i in range(numLayers):
                self.decimations.append([0,0,0])
            
            # Precompute decimations for each layer.
            
            for iLayer in range(1, numLayers):
                # For each parent layer...
                self.computeDecimations( self.decimations[ iLayer ] , iLayer )
            # Layer 0 is strictly a child (i.e. has no children), so has no decimations.
            # Assign the values with useless nonsense to make this more obvious.
            #self.decimations[0] = [0,0,0]


        '''! \brief Get indices of minimal cell in child layer of cluster represented by specified cell in parent layer.

            Each cell in a parent layer represents a grid cluster of typically 8 cells
            in the child layer.  This routine calculates the index of the "minimal"
            cell in the child layer grid cluster, i.e. the cell in the child layer
            which corresponds to minimum corner cell of the grid cluster represented
            by the cell in the parent layer with the specified index.

            The cells in the child layer that belong to the same grid cluster would
            be visited by this code:

            \verbatim

                int i[3] ; // i is the increment past the minimum corner cell in the grid cluster.
                int j[3] ; // j indexes into the child layer.
                for( i[2] = 0 ; i[2] <= decimations[2] ; ++ i[2] )
                {
                    j[2] = i[2] + clusterMinIndices[2] ;
                    for( i[1] = 0 ; i[1] <= decimations[1] ; ++ i[1] )
                    {
                        j[1] = i[1] + clusterMinIndices[1] ;
                        for( i[0] = 0 ; i[0] <= decimations[0] ; ++ i[0] )
                        {
                            j[0] = i[0] + clusterMinIndices[0] ;
                            // Use j to index into child layer.
                        }
                    }
                }

            \endverbatim

            \param clusterMinIndices - (out) index of minimal cell in child layer grid cluster represented by given parent cell

            \param decimations - (in) ratios of dimensions of child layer to its parent, for each axis.
                    This must be the same as the result of calling GetDecimations for the intended parent layer.

            \param indicesOfParentCell - (in) index of cell in parent layer.

            \see GetDecimations

        '''
        def getChildClusterMinCornerIndex( self , decimations , indicesOfParentCell ):
            clusterMinIndices = [0,0,0]
            clusterMinIndices[ 0 ] = indicesOfParentCell[ 0 ] * decimations[ 0 ]
            clusterMinIndices[ 1 ] = indicesOfParentCell[ 1 ] * decimations[ 1 ]
            clusterMinIndices[ 2 ] = indicesOfParentCell[ 2 ] * decimations[ 2 ]
            return clusterMinIndices
        

        def getDecimations(self, iParentLayer ):
            return self.decimations[ iParentLayer ]

        def dumpSelf(self):
            count = 0
            for layer in self:
                dumpFile = open( "/tmp/layer" + str(count) + ".dmp", "w" )
                dump = layer.dumpCells()
                count+=1
                dumpFile.write(dump)
                dumpFile.close()