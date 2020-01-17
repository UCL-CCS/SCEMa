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
		double 	update_strain[6];
		double	update_stress[6];
	};

	struct ScaleBridgingData
	{
		std::vector<QP>	update_list;
	};

	MPI_Datatype MPI_QP;
	void create_qp_mpi_datatype()
	{
    MPI_Type_contiguous(sizeof(QP), MPI_BYTE, &MPI_QP);
    MPI_Type_commit(&MPI_QP);
	}
}


#endif 

