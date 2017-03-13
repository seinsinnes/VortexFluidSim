module grid

	real*8, dimension (:,:,:,:), allocatable :: gridContents
	integer :: gridSizeX,gridSizeY,gridSizeZ
	real*8, dimension (3) :: cellsPerExtent

	contains
	function setGridContents(gridCont, gridS_x,gridS_y,gridS_z)
		integer, intent(in) :: gridS_x,gridS_y,gridS_z
		real*8, dimension(gridS_x,gridS_y,gridS_z,3),intent(in) :: gridCont
		
		print *,"in"
		print *,gridCont(4,5,6,1:3)
		if(.not.(allocated(gridContents).and.gridSizeX == gridS_x.and.gridSizeY == gridS_y.and.gridSizeZ == gridS_z)) then
				if (allocated(gridContents)) then
					deallocate(gridContents)
				end if
				gridSizeX = gridS_x
				gridSizeY = gridS_y
				gridSizeZ = gridS_z
				allocate(gridContents(gridS_x,gridS_y, gridS_z,3))
		end if

		gridContents = gridCont

	end function setGridContents

	function setCellsPerExtent( cpe)
		real*8, intent(in) :: cpe
		cellsPerExtent = cpe
	end function setCellsPerExtent
		

	pure function getCellContents( indices )
		integer, dimension(3), intent(in) :: indices
		real*8, dimension(3) :: getCellContents
		getCellContents = gridContents(indices(1),indices(2),indices(3),1:3)
	end function getCellContents


	pure function interpolate( vPosition, indices,  vMinCorner)
		real*8, dimension(3), intent(in) :: vPosition, vMinCorner
		integer, dimension(3), intent(in) :: indices
		integer, dimension(3) ::  x0y0z0,x1y0z0, &
					x1y1z0,x1y1z1,x0y1z0,x0y1z1,x1y0z1,x0y0z1
		real*8, dimension(3) :: vDiff,tween,oneMinusTween,interpolate
		
		! Relative location of position within its containing grid cell.
		vDiff         = vPosition - vMinCorner 
            !#unsigned        indices[3] ; // Indices of grid cell containing position.
            !indices = self.indicesOfPosition( vPosition )
            !#print "i", indices
            !#print vPosition
            !#Vec3            vMinCorner ;
            !vMinCorner = self.positionFromIndices( indices )
            !#const unsigned  offsetX0Y0Z0 = OffsetFromIndices( indices ) ;
	    !print *,"calc"
            !print *,vDiff
	    !print *,cellsPerExtent
            tween         = vDiff * cellsPerExtent
	    !print *,tween
            oneMinusTween = (/ 1.0, 1.0, 1.0 /) - tween

            !#const unsigned  numXY         = GetNumPoints( 0 ) * GetNumPoints( 1 ) ;
            !#const unsigned  offsetX1Y0Z0  = offsetX0Y0Z0 + 1 ;
            x0y0z0 = indices
            x1y0z0 = x0y0z0 + (/1,0,0/)
            !#const unsigned  offsetX0Y1Z0  = offsetX0Y0Z0 + GetNumPoints(0) ;
            x0y1z0 = x0y0z0 + (/0,1,0/)
            !#const unsigned  offsetX1Y1Z0  = offsetX0Y0Z0 + GetNumPoints(0) + 1 ;
            x1y1z0 = x0y0z0 + (/1,1,0/)
            !#const unsigned  offsetX0Y0Z1  = offsetX0Y0Z0 + numXY ;
            x0y0z1 = x0y0z0 + (/0,0,1/)
            
            !#const unsigned  offsetX1Y0Z1  = offsetX0Y0Z0 + numXY + 1 ;
            x1y0z1 = x0y0z0 + (/1,0,1/)
            !#const unsigned  offsetX0Y1Z1  = offsetX0Y0Z0 + numXY + GetNumPoints(0) ;
            x0y1z1 = x0y0z0 + (/0,1,1/)
            !#const unsigned  offsetX1Y1Z1  = offsetX0Y0Z0 + numXY + GetNumPoints(0) + 1 ;
            x1y1z1 = x0y0z0 + (/1,1,1/)

            interpolate = oneMinusTween(1) * oneMinusTween(2) * oneMinusTween(3) * getCellContents(x0y0z0) &
            +         tween(1) * oneMinusTween(2) * oneMinusTween(3) * getCellContents(x1y0z0 ) &
            + oneMinusTween(1) *         tween(2) * oneMinusTween(3) * getCellContents(x0y1z0 ) &
            +         tween(1) *         tween(2) * oneMinusTween(3) * getCellContents(x1y1z0 ) &
            + oneMinusTween(1) * oneMinusTween(2) *         tween(3) * getCellContents(x0y0z1 ) &
            +         tween(1) * oneMinusTween(2) *         tween(3) * getCellContents(x1y0z1 ) &
            + oneMinusTween(1) *         tween(2) *         tween(3) * getCellContents(x0y1z1 ) &
            +         tween(1) *         tween(2) *         tween(3) * getCellContents(x1y1z1 )
       end function interpolate
end module 
