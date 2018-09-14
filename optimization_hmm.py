import numpy as np
import sys
import os
import math
import imp
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import json
from scipy.optimize import curve_fit

# Decreasing exponential (might want to introduce Amdahl's law for latency not speedup)
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def compute_p_optimum(data, mode='efficiency',cpn=28):
    if mode =='time-to-completion':
        # Pmax as the number of nodes which yields the quickest result
        pmax = math.ceil(data[data.argmin(axis=0)[1],0]/cpn)
    elif mode=='efficiency':
        # The factor between the speedup at which pmax is chosen and the initial speedup
        relative_performance_factor = 0.9
        # Compute regression fit of the performance data using a inverse exponential
        xdata = data[:,0]
        ydata = data[:,1]
        popt, pcov = curve_fit(func, xdata, ydata)
        yfit = func(xdata, *popt)
        # Compute the instantaneous slope of the fitted exponential
        slope = -np.gradient(yfit)/np.gradient(xdata)
        # Set Pmax as the number of nodes for which the slope becomes less than
        # a given percentage of the initial slope
        for i in range(slope.size):
            cslope = slope[i]
            if cslope < relative_performance_factor*slope[0]:
                pmax = xdata[i]/cpn
    return pmax

# to access database: source /home/plgrid-groups/plggcompat/anaconda2/bin/activate performance-database
# Input list: "number_of_cells" "upstrain_files_directory" "minimum_nodes_job"
# execute as: python optimization_hmm.py ./macroscale_state/out 1 28 1-1 ./nanoscale_state/out ./nanoscale_log/tmp ./nanoscale_state/out/job_list_md.json 
if __name__ == '__main__':

    # static parameters
    rescale_from_time_prediction = False
    pmax_from_db = False
    negl_strain_tsh = 1.0e-10
    cores_per_nodes = 28

    # input parameters
    macrostatelocout = sys.argv[1]
    Pmin = int(sys.argv[2]) # Set the minimum number of nodes assigned to an microjob
    nreplicas = int(sys.argv[3])
    time_id = sys.argv[4]
    nanostatelocout = sys.argv[5]
    nanologloctmp = sys.argv[6]
    joblistfile = sys.argv[7]

    # Need to parse the material id list (material id for each cell)
    cell_list = []
    jobs_count = 0
    matcell_file = macrostatelocout+"/cell_id_mat.list"
    with open(matcell_file) as fp:
        for line in fp:
            cell = {}
            cell['id'] = line.split(" ")[0]
            cell['mat'] = line.split(" ")[1][:-1]

            # Load the strain tensor for all cells in the mesh, a null vector is loaded if the file is not found
            try:
                with open('{}/last.{}.upstrain'.format(macrostatelocout, cell['id']), 'r') as fd:
                    cell['strain_tensor'] = np.array([float(fd.readline()) for _ in range(6)])
            except IOError as e:
                # print( "Error: {} - Could not find strain file for cell {}".format(e, cell['id']))
                cell['strain_tensor'] = np.zeros(6)

            cell['strain_norm'] = np.linalg.norm(cell['strain_tensor'])

            # Only keep cells with a non-negligible strain to compute otherwise useless
            if cell['strain_norm'] > negl_strain_tsh:
                cell_list.append(cell)
                # increase jobs_count to set cell_id as line number of the matlist file
                jobs_count+=1

    # If list of jobs is 0 exit
    if jobs_count == 0:
        print(0)

    # Else produce the job list and return the total number of nodes
    else:
        Create instance of the Performance database
        database = imp.load_source('database', '/home/plgrid-groups/plggcompat/Common/COMPAT-Patterns-Software/OptimisationPart/database_upload.py')
        engine = create_engine('mysql://performance_update:password@129.187.255.55/performance')
        Session = sessionmaker(bind=engine)
        session = Session()
        run_ids_by_material = [instance.id for instance in session.query(database.Run).filter(database.Run.application == 'MATERIALS')]

        # Compute Pmax as the number of nodes for which the latency (time/workload)
        # is not linearly increasing with the number of nodes anymore
        if pmax_from_db:
            kernelperf_for_materials = np.array([{'id':instance.id, 'kernel_id':instance.kernel_id, 'corecount':instance.cores, 'runtime':instance.runtime} for instance in session.query(database.KernelPerf).filter(database.KernelPerf.run_id.in_(run_ids_by_material))])
            unique_kernel_id = np.unique([d['kernel_id'] for d in kernelperf_for_materials])
            eps_by_kernel_id = {}
            for instance in session.query(database.Kernel).filter(database.Kernel.id.in_(unique_kernel_id)):
                parameters = json.loads(instance.config_parameters)
                try:
                    eps_by_kernel_id[instance.id] = np.linalg.norm([parameters['strain_tensor'][val] for val in parameters['strain_tensor']])
                except KeyError:
                    pass
            ## keeping only runs with a strain norm above neg_strain_tsh, less is barely significant and the runtime might be irrelevant
            kernelperf_corecount_latency = np.array([[instance['corecount'], instance['runtime']/eps_by_kernel_id[instance['kernel_id']]] \
                for instance in kernelperf_for_materials if eps_by_kernel_id[instance['kernel_id']] > negl_strain_tsh])

            Pmax = compute_p_optimum(data=kernelperf_corecount_latency, mode='efficiency', cpn=cores_per_nodes)
        # Use a default variable (from arbitrary choice)
        else:
            Pmax = 20

        # Compute a scaling factor that will be used to determine the number of nodes allocated to each cell job
        # accounting on the all the jobs that have to be run.
        # Either computed using the predicted execution time from performance measurements of the DB or simply using
        # the norm of the strain tensor that will be applied during the job
        if rescale_from_time_prediction:
            # Query execution time Texec for each cell for min node
            texec = []
            for cell in cell_list:
                kernelperf_id_by_run_ids_and_corecount = [instance[0] for instance in session.query(database.KernelPerf.kernel_id).distinct(database.KernelPerf.kernel_id).filter(database.KernelPerf.run_id.in_(run_ids_by_material) & database.KernelPerf.cores == Pmin*cores_per_nodes)]
                kernel_id_by_kernelperf_id_and_epsilon = []
                for instance in session.query(database.Kernel).filter(database.Kernel.id.in_(kernelperf_id_by_run_ids_and_corecount)):
                    parameters = json.loads(instance.config_parameters)
                    try:
                        # might have to allow some tolerance on that condition,
                        # as the exact same strain might not have been computed already
                        if float(parameters['strain_norm']) == cell['strain_norm']:
                            kernel_id_by_kernelperf_id_and_epsilon.append(instance.id)
                    except KeyError:
                        pass
                kernelperf_time_by_kernel_id = np.array([instance.runtime for instance in session.query(database.KernelPerf).filter(database.KernelPerf.id.in_(kernel_id_by_kernelperf_id_and_epsilon))])
                cell['texec'] = kernelperf_time_by_kernel_id.mean()

            # Compute the rescaling factor alpha such as (Pmax-Pmin)/(Texec,max - Texec,min)
            texec_min = min([cell['texec'] for cell in cell_list])
            texec_max = max([cell['texec'] for cell in cell_list])
            if (texec_max - texec_min)/texec_min > 1.0e-5:
                sfact = (Pmax-Pmin)/(texec_max - texec_min)
            else:
                # basically if all cells have the same execution time, just assign to all have them
                # half of the maximum of resources
                sfact = (Pmax-Pmin)/(texec_max*2)

        else:
            # Compute the rescaling factor alpha such as (Pmax-Pmin)/(eps.norm,max - eps.norm,min)
            strain_min = min([cell['strain_norm'] for cell in cell_list])
            strain_max = max([cell['strain_norm'] for cell in cell_list])
            if (strain_max - strain_min)/strain_min > 1.0e-5:
                sfact = (Pmax-Pmin)/(strain_max - strain_min)
            else:
                # basically if all cells have the same applied strain, just assign to all have them
                # half of the maximum of resources
                sfact = (Pmax-Pmin)/(strain_max*2)

        # Compute the number of nodes allocated to each microjob P such as alpha*T and the total number of nodes
        ptot = 0
        for cell in cell_list:
            if rescale_from_time_prediction:
                cell['p'] = round(cell['texec']*sfact)
            else:
                cell['p'] = round(cell['strain_norm']*sfact)
            ptot += cell['p']

        # Sort cell_ids in decreasing order of P
        cell_list_sorted = sorted(cell_list, key=lambda k: k['p'], reverse=True)

        # Write the JSON file for the PJM
        json_job_list = []
        top_json = {}
        top_json['request'] = 'submit'
        top_json['jobs'] = []
        for cell in cell_list_sorted:
            cell_job_dict = {}
            cell_job_dict['name'] = 'mdrun_cell'+str(cell['id'])+'_repl${it}'
            cell_job_dict['iterate'] = [1, nreplicas]
            cell_job_dict['execution'] = {}
            cell_job_dict['execution']['exec'] = 'bash'
            cell_job_dict['execution']['args'] = [nanostatelocout+"/"+"bash_cell"+str(cell['id'])+"_repl${it}.sh"]
            cell_job_dict['execution']['stdout'] = nanologloctmp+"/"+time_id+"."+str(cell['id'])+"."+str(cell['mat'])+"_${it}"+'/${jname}.stdout'
            cell_job_dict['execution']['stderr'] = nanologloctmp+"/"+time_id+"."+str(cell['id'])+"."+str(cell['mat'])+"_${it}"+'/${jname}.stderr'
            cell_job_dict['resources'] = {}
            cell_job_dict['resources']['numNodes'] = {}
            cell_job_dict['resources']['numNodes']['exact'] = cell['p']
            top_json['jobs'].append(cell_job_dict)
        json_job_list.append(top_json)

        bottom_json = {}
        bottom_json['request'] = 'control'
        bottom_json['command'] = 'finishAfterAllTasksDone'
        json_job_list.append(bottom_json)

        json = json.dumps(json_job_list)
        f = open(joblistfile,"w")
        f.write(json)
        f.close()

        # Returning the total number of nodes to the main app
        print(ptot)
