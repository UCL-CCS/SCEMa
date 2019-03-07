#ifndef SCALE_BRIDGING_DATA_H
#define SCALE_BRIDGING_DATA_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>
#include <fstream>
#include <math.h>
namespace HMM { 
	
	struct QP // quadradture point
	{
		int 	id;
		int 	material;
	};

	struct ScaleBridgingData
	{
		std::vector<QP>	update_list;
	};

}


#endif 

