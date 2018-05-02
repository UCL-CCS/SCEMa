import zmq
import time

from qcg.appscheduler.api.manager import Manager
from qcg.appscheduler.api.job import Jobs

# switch on debugging (by default in api.log file)
m = Manager( cfg = { 'log_level': 'DEBUG' } )

# get available resources
print("available resources:\n%s\n" % str(m.resources()))

# submit jobs and save their names in 'ids' list
ids = m.submit(Jobs().
        add( 'env', { 'exec': '/bin/env', 'stdout': 'env.stdout', 'nodes': { 'exact': 1 }, 'cores': { 'exact': 2 } } ).
        add( 'host', { 'exec': '/bin/hostname', 'args': [ '--fqdn' ], 'stdout': 'host.stdout' } )
        )

# list submited jobs
print("submited jobs:\n%s\n" % str(m.list(ids)))

# wait until submited jobs finish
m.wait4(ids)

# get detailed information about submited and finished jobs
print("jobs details:\n%s\n" % str(m.info(ids)))

