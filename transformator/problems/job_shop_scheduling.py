from datetime import date
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px

from transformator.problems.problem import Problem


class JobShopScheduling(Problem):
    def __init__(self, cfg, jobs, T=0):
        # jobs = {job_number:[operations]}
        # operation =[machine, duration]
        self.jobs = jobs
        self.T = T
        self.a = len(jobs)

    def gen_qubo_matrix(self):
        n_machines = 0
        for key, job in self.jobs.items():
            for operation in job:
                if operation[0] >= n_machines:
                    n_machines = operation[0]
        n_machines += 1
        
        machines_ops_map = {n: [] for n in range(n_machines)}
        processing_time = {}
        jobs_ops_map = {}
        ops_jobs_map = {}

        index = 0
        for key, job in self.jobs.items():
            operations = []
            for operation in job:
                processing_time.update({index: operation[1]})
                operations.append(index)
                machines_ops_map[operation[0]].append(index)
                ops_jobs_map.update({index: key})
                index += 1
            jobs_ops_map.update({key: operations})

        n_machines = len(machines_ops_map)
        n_operations = len(ops_jobs_map)

        # T is a strict upper bound where all jobs should be finished.
        # In the Paper, T is the latest time an operation can be scheduled, so
        # for long operations, they finish after deadline
        if self.T > 0:
            T = self.T
        else:
            # If we dont have a fixed time, take the minimum time and add a
            # buffer of 50% (1.5) to avoid overlappings
            # This works not very well, so the recommendation is always
            # to give a fixed time with the params
            sum_durations = sum(processing_time.values())
            T = min(int(np.ceil((sum_durations*1.5)/n_machines)), sum_durations)

        V = T * n_operations * n_machines

        # Init weights for penalty terms. In the paper by 1qubit it is only
        # required that they are greater 0.
        alpha = 1
        beta = 1
        gamma = 1

        Q = np.zeros((V, V))

        # Constraints from microsoft tutorial 'Solve a job shop scheduling
        # optimization problem' by using Azure Quantum.
        # https://docs.microsoft.com/en-us/learn/modules/solve-job-shop-optimization-azure-quantum/  # noqa

        # Precedence constraint.
        # Loop through all jobs:
        for ops in jobs_ops_map.values():
            # Loop through all operations in this job:
            for i in range(len(ops) - 1):
                for t in range(0, T):
                    # Loop over times that would violate the constraint:
                    for s in range(0, min(t + processing_time[ops[i]], T)):
                        # Assign penalty
                        Q[ops[i]*T+t, (ops[i+1])*T+s] += alpha

        # Operation-once constraint.
        # Here we changed some details from the paper to make sure, that every
        # job finishes before deadline.
        ops_index = 0
        for ops in self.jobs.values():
            for op in ops:
                # Take duration and start every job at a time, so that they can
                # finish before deadline.
                # NOTE: Adding '-duration' is a modification to the paper.
                duration = op[1]
                for t in range(T-duration+1):
                    # - x - y terms
                    Q[ops_index*T+t][ops_index*T+t] -= beta

                    # + 2xy term
                    # Loop through all other start times for the same job
                    # to get the cross terms
                    for s in range(t+1, T):
                        Q[ops_index*T+t, ops_index*T+s] += 2*beta

                # If job starts too close to deadline, add penalty.
                # NOTE: This is also a modification to the paper.
                for t in range(T-duration+1, T):
                    Q[ops_index*T+t][ops_index*T+t] += beta

                ops_index += 1

        # No-overlap constraint.
        for ops in machines_ops_map.values():
            # Loop over each operation i requiring this machine
            for i in ops:
                # Loop over each operation k requiring this machine 
                for k in ops:
                    # Loop over simulation time
                    for t in range(T):
                        # When i != k (when scheduling two different
                        # operations).
                        # t = s meaning two operations are scheduled to start
                        # at the same time on the same machine.
                        if i > k:
                            # We dont need that, because it is in next loop
                            Q[k*T+t, i*T+t] += gamma

                            # Add penalty when operation runtimes overlap
                            for s in range(t, min(t + processing_time[i], T)):
                                Q[k*T+s, i*T+t] += gamma 

                            # If operations are in the same job, penalize for
                            # the extra time 0 -> t (operations scheduled out
                            # of order)
                            if ops_jobs_map[i] == ops_jobs_map[k]:
                                for s in range(0, t):
                                    Q[k*T+t, i*T+s] += gamma 

                        # We dont need that, because its doubled
                        if i < k:
                            Q[i*T+t, k*T+t] += gamma

                            # Add penalty when operation runtimes overlap
                            for s in range(t, min(t + processing_time[i], T)):
                                Q[i*T+t, k*T+s] += gamma 

                            # If operations are in the same job, penalize for
                            # the extra time 0 -> t (operations scheduled out
                            # of order)
                            if ops_jobs_map[i] == ops_jobs_map[k]:
                                for s in range(0, t):
                                    Q[i*T+t, k*T+s] += gamma

        return 1/2*(Q+Q.transpose())

    @classmethod
    def gen_problems(
            self,
            cfg,
            n_problems,
            n_jobs=4,
            n_machines=1,
            max_duration=2,
            operations_per_job=[1,3],
            T=0,
            **kwargs):
        problems = []
        for _ in range(n_problems):
            jobs = {}

            if operations_per_job[0] == operations_per_job[1]:
                jobs_list = np.full(n_jobs,operations_per_job[0])
            else:
                jobs_list = np.random.randint(
                    low=operations_per_job[0],
                    high=operations_per_job[1]+1,
                    size=(n_jobs,)
                ).tolist()

            operations = []

            for job, n_operations in enumerate(jobs_list):
                operations = []
                for _ in range(n_operations):
                    machine = np.random.randint(n_machines, size=1)[0]
                    if max_duration == 1:
                        duration = 1
                    else:
                        duration = np.random.randint(
                            1,
                            max_duration+1,
                            size=1
                        )[0]
                    operations.append([machine, duration])
                jobs.update({job:operations})
            problems.append({"jobs": jobs, "T": T})
        return problems

    @classmethod
    def get_solution_dict(self, problem, solution):
        # start_jobs = {job: schedule}
        # schedule = {starttime: operation}
        # operation =[machine, duration]

        n_operations = 0
        n_machines = 1
        for _ , ops in problem.items():
            for op in ops:
                n_operations += 1
                if op[0] >= n_machines:
                    n_machines = op[0] + 1
        T = int(len(solution) / (n_operations * n_machines))

        start_operations = {o: -1 for o in range(n_operations)}
        for t in range(T):
            for o in range(n_operations):
                if solution[o*T + t] == 1:
                    start_operations.update({o:t})
        
        start_jobs = {}
        operation_index = 0
        for job, operations in problem.items():
            ops_start_dict = {}
            for operation in operations:
                ops_start_dict.update(
                    {start_operations[operation_index]: operation}
                )
                operation_index += 1
            start_jobs.update({job: ops_start_dict})

        return start_jobs

    @classmethod
    def plot_solution_dict(self, solution, name="gantt.png"):
        data = {}
        index = 0
        init_date = date(2021, 1, 1)
        day = timedelta(days=1)
        for job_n, job in solution.items():
            for start, operation in job.items():
                if start != -1:
                    start_date = init_date + day * start
                    end_date = start_date + day * (operation[1])
                    m = operation[0]
                    maschine = "machine" + str(m)
                    data[index] = {
                        "job": str(job_n),
                        "machine": maschine,
                        "start": start_date, 
                        "finish": end_date }
                    index += 1

        sol_len = len(data)

        l1 = [ dict(
                start=data[i]["start"], 
                finish=data[i]["finish"], 
                machine=data[i]["machine"], 
                job=data[i]["job"]
                ) for i in range(sol_len)]
        df = pd.DataFrame(l1)

        colors = {  
                    'maschine0':  'rgb(52, 152, 219)',
                    'maschine1':  'rgb(231, 76, 60)',
                    'maschine2':  'rgb(46, 204, 113)',
                    'maschine3':  'rgb(241, 196, 15)',
                    'maschine4':  'rgb(155, 89, 182)',
                    'maschine5':  'rgb(52, 73, 94)',
                    'maschine6':  'rgb(230, 126, 34)',
                    'maschine7':  'rgb(22, 160, 133)'
                }

        fig = px.timeline(
            df, 
            x_start="start", 
            x_end="finish", 
            y="job", 
            color="machine", 
            color_discrete_map=colors )
        fig.write_image(name)