#ifndef READ_WRITE_H
#define READ_WRITE_H

#include "boost/property_tree/ptree.hpp"

#include <deal.II/base/symmetric_tensor.h>

using namespace dealii;

void bptree_print(boost::property_tree::ptree const& pt)
{
	using boost::property_tree::ptree;
	ptree::const_iterator end = pt.end();
	for (ptree::const_iterator it = pt.begin(); it != end; ++it) {
		std::cout << it->first << ": " << it->second.get_value<std::string>() << std::endl;
		bptree_print(it->second);
	}
}

std::string bptree_read(boost::property_tree::ptree const& pt, std::string key)
{
	std::string value = "NULL";
	using boost::property_tree::ptree;
	ptree::const_iterator end = pt.end();
	for (ptree::const_iterator it = pt.begin(); it != end; ++it) {
		if(it->first==key)
			value = it->second.get_value<std::string>();
	}
	return value;
}

std::string bptree_read(boost::property_tree::ptree const& pt, std::string key1, std::string key2)
{
	std::string value = "NULL";
	using boost::property_tree::ptree;
	ptree::const_iterator end = pt.end();
	for (ptree::const_iterator it = pt.begin(); it != end; ++it) {
		if(it->first==key1){
			value = bptree_read(it->second, key2);
		}
	}
	return value;
}

std::string bptree_read(boost::property_tree::ptree const& pt, std::string key1, std::string key2, std::string key3)
{
	std::string value = "NULL";
	using boost::property_tree::ptree;
	ptree::const_iterator end = pt.end();
	for (ptree::const_iterator it = pt.begin(); it != end; ++it) {
		if(it->first==key1){
			value = bptree_read(it->second, key2, key3);
		}
	}
	return value;
}

boost::property_tree::ptree get_subbptree(boost::property_tree::ptree const& pt, std::string key1)
{
	boost::property_tree::ptree value;
	using boost::property_tree::ptree;
	ptree::const_iterator end = pt.end();
	for (ptree::const_iterator it = pt.begin(); it != end; ++it) {
		if(it->first==key1){
			value = it->second;
		}
	}
	return value;
}


bool file_exists(std::string file) {
	struct stat buf;
	return (stat(file.c_str(), &buf) == 0);
}


bool file_exists(const char* file) {
	struct stat buf;
	return (stat(file, &buf) == 0);
}


template <int dim>
inline
void
read_tensor (const char *filename, double &tensor)
{
	std::ifstream ifile;

	ifile.open (filename);
	if (ifile.is_open())
	{
		char line[1024];
		if(ifile.getline(line, sizeof(line)))
			tensor = std::strtod(line, NULL);
		ifile.close();
	}
	else std::cout << "Unable to open" << filename << " to read it" << std::endl;
}

template <int dim>
inline
void
read_tensor (const char *filename, Tensor<1,dim> &tensor)
{
	std::ifstream ifile;

	ifile.open (filename);
	if (ifile.is_open())
	{
		for(unsigned int k=0;k<dim;k++)
			{
				char line[1024];
				if(ifile.getline(line, sizeof(line)))
					tensor[k] = std::strtod(line, NULL);
			}
		ifile.close();
	}
	else std::cout << "Unable to open" << filename << " to read it" << std::endl;
}

template <int dim>
inline
bool
read_tensor (const char *filename, SymmetricTensor<2,dim> &tensor)
{
	std::ifstream ifile;

	bool load_ok = false;

	ifile.open (filename);
	if (ifile.is_open())
	{
		load_ok = true;
		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
			{
				char line[1024];
				if(ifile.getline(line, sizeof(line)))
					tensor[k][l] = std::strtod(line, NULL);
			}
		ifile.close();
	}
	else std::cout << "Unable to open" << filename << " to read it" << std::endl;
return load_ok;
}

template <int dim>
inline
void
read_tensor (const char *filename, SymmetricTensor<4,dim> &tensor)
{
	std::ifstream ifile;

	ifile.open (filename);
	if (ifile.is_open())
	{
		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
				for(unsigned int m=0;m<dim;m++)
					for(unsigned int n=m;n<dim;n++)
					{
						char line[1024];
						if(ifile.getline(line, sizeof(line)))
							tensor[k][l][m][n]= std::strtod(line, NULL);
					}
		ifile.close();
	}
	else std::cout << "Unable to open" << filename << " to read it..." << std::endl;
}

template <int dim>
inline
void
write_tensor (const char *filename, double &tensor)
{
	std::ofstream ofile;

	ofile.open (filename);
	if (ofile.is_open())
	{
		ofile << std::setprecision(16) << tensor << std::endl;
		ofile.close();
	}
	else std::cout << "Unable to open" << filename << " to write in it" << std::endl;
}

template <int dim>
inline
void
write_tensor (const char *filename, Tensor<1,dim> &tensor)
{
	std::ofstream ofile;

	ofile.open (filename);
	if (ofile.is_open())
	{
		for(unsigned int k=0;k<dim;k++)
				//std::cout << std::setprecision(16) << tensor[k][l] << std::endl;
				ofile << std::setprecision(16) << tensor[k] << std::endl;
		ofile.close();
	}
	else std::cout << "Unable to open" << filename << " to write in it" << std::endl;
}

template <int dim>
inline
void
write_tensor (const char *filename, SymmetricTensor<2,dim> &tensor)
{
	std::ofstream ofile;

	ofile.open (filename);
	if (ofile.is_open())
	{
		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
				//std::cout << std::setprecision(16) << tensor[k][l] << std::endl;
				ofile << std::setprecision(16) << tensor[k][l] << std::endl;
		ofile.close();
	}
	else std::cout << "Unable to open" << filename << " to write in it" << std::endl;
}

template <int dim>
inline
void
write_tensor (const char *filename, SymmetricTensor<4,dim> &tensor)
{
	std::ofstream ofile;

	ofile.open (filename);
	if (ofile.is_open())
	{
		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
				for(unsigned int m=0;m<dim;m++)
					for(unsigned int n=m;n<dim;n++)
						ofile << std::setprecision(16) << tensor[k][l][m][n] << std::endl;
		ofile.close();
	}
	else std::cout << "Unable to open" << filename << " to write in it" << std::endl;
}


#endif
