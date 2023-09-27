MODULE computeVel

	!integer, parameter :: dp = selected_real_kind(15,307)
	real*8, parameter :: fourPi = (4.0 * 3.1415926535897932384626433832795)
	real*8, parameter :: oneOverFourPi       = 1.0 / fourPi
	real*8 :: sAvoidSingularity
	real*8, dimension (:,:,:) , allocatable :: vortonInfoList
	integer :: vortonListLen
	contains

	pure function crossProduct(a, b)
  		real*8, dimension(3) :: crossProduct
  		real*8, dimension(3), intent(in) :: a, b
  		crossProduct(1) = a(2) * b(3) - a(3) * b(2)
  		crossProduct(2) = a(3) * b(1) - a(1) * b(3)
  		crossProduct(3) = a(1) * b(2) - a(2) * b(1)
	end function crossProduct

	function setVortonInfoList( vInfo, vListLen )
		integer, intent(in) :: vListLen
		real*8, dimension(vListLen,3, 3),intent(in) :: vInfo
		if(.not.allocated(vortonInfoList)) allocate(vortonInfoList(vListLen,3,3))
		vortonInfoList = vInfo
		vortonListLen = vListLen
		sAvoidSingularity = tiny(fourPi)**(1.0/3.0)
	end function setVortonInfoList
		

	pure function computeVelocityBruteForce( vPosition )
		
		real*8, dimension(3), intent(in) :: vPosition
		
		real*8, dimension(vortonListLen,3) :: velocities
		real*8, dimension(3) :: computeVelocityBruteForce

		forall (i = 1:vortonListLen) velocities(i,1:3) = accumulateVelocity( vPosition, &
				vortonInfoList(i,1:3,1:3))

		!do i = 1,vortonListLen 
		!	velocities(i,1:3) = accumulateVelocity( vPosition, &
		!		vortonInfoList(i,1:3,1:3))
		!end do

		computeVelocityBruteForce = sum(velocities,dim=1)
		
	end function computeVelocityBruteForce

	pure function accumulateVelocity( vPosQuery, vortonInfo )
		real*8, dimension(3,3), intent(in) :: vortonInfo
		real*8, dimension(3), intent(in) :: vPosQuery
		real*8, dimension(3) :: accumulateVelocity
		
		real*8, dimension(3) :: vNeighborToSelf
		                                                                                                 
		real*8 :: dist2, oneOverDist, radius2

		!double vcProdVortNeighbour[3];
	
		!printf(" %f \n", vortPos[0]);
		vNeighborToSelf =  vPosQuery - (vortonInfo(1,1:3))
		!vNeighborToSelf     = vPosQuery - vortonInfo[0]                                       
      		radius2             = vortonInfo(3,1) * vortonInfo(3,1)
        	dist2               = sum(vNeighborToSelf**2) + sAvoidSingularity
	              
        	oneOverDist         = 1.0 / sqrt( dist2 )
	 
        	!vNeighborToSelfDir  = vNeighborToSelf * oneOverDist
        	!If the reciprocal law is used everywhere then when 2 vortices get close, they tend to jettison. '''
        	!Mitigate this by using a linear law when 2 vortices get close to each other.*/
        	!print vVelocity
        	if (dist2 < radius2) then
			!Inside vortex core
            		distLaw = ( oneOverDist / radius2 )     
		else
			!Outside vortex core                             
            		distLaw = ( oneOverDist / dist2 )
		end if
		!printf("%f\n",OneOverFourPi);
        	!vVelocity +=  OneOverFourPi * ( 8.0 * radius2 * mRadius ) * numpy.cross(mVorticity, vNeighborToSelf) * distLaw
		!crossProduct(vcProdVortNeighbour, vortVort, vNeighborToSelf);
		accumulateVelocity = oneOverFourPi * 8.0 * radius2 * vortonInfo(3,1) * distLaw * crossProduct(vortonInfo(2,1:3), vNeighborToSelf)
		!scalarProduct(velocityTemp, vcProdVortNeighbour, OneOverFourPi * ( 8.0f * radius2 * vortRadius )*distLaw);
        	!vVelocity =  OneOverFourPi * ( 8.0 * radius2 * vortRadius ) * crossProduct(vortonInfo[1], vNeighborToSelf) * distLaw   
        	!print vVelocity
		!print *, accumulateVelocity
		!addVectors3D(vVelocity,vVelocity, velocityTemp);
	end function accumulateVelocity
END MODULE

