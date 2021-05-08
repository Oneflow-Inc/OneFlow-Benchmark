import os
import csv
import time


def local_run(log_file, cmd, nsys='', python_cmd='python3'):
    cmd += f' | tee {log_file}.log'
    if nsys != '':
        cmd = cmd.replace(python_cmd, f'{nsys}--output={log_file} {python_cmd}')
    print('executing:', cmd)
    start = time.time()
    os.system(cmd)
    duration = time.time() - start
    print('duration: ', duration)


def remote_run(log_file, cmd, host, port=None, python_cmd='python3', workspace='/workspace'):
    ''' nohup run '''
    ssh = 'ssh '
    if port is not None:
        ssh += f'-p{port} '
    ssh = ssh + host + ' "'
    ssh += f'cd {workspace}; '
    cmd = cmd.replace(python_cmd, f'nohup {python_cmd}')
    cmd += f' 1>{log_file}.log 2>&1 </dev/null &"'
    cmd = ssh + cmd
    print('executing:', cmd)
    # cmd = ssh + 'ls'
    os.system(cmd)

    
def gpt_run(args, env, test_case, port=None, nsys='', python_cmd='python3', workspace='/workspace'):
    # print(test_case)
    def dict_2_str(d, prefix=''):
        return "".join(["{}{}={} ".format(prefix, k, v) for k, v in d.items()])

    cmd = dict_2_str(env)
    cmd += f"{python_cmd} -m oneflow_gpt.training "
    cmd += dict_2_str(args, '--')
    cmd += "--checkpoint-activations"

    num_nodes = int(args['num-nodes'])
    ips = args['node-ips'].split(',')
    if int(num_nodes) > 1:
        assert len(ips) >= num_nodes
        for i in range(1, num_nodes): #ignore master node
            log_file = os.path.join(args['log'], f'{test_case}_{ips[i]}')
            remote_run(log_file, cmd, ips[i], port=port, python_cmd=python_cmd, workspace=workspace)
    log_file = os.path.join(args['log'], f'{test_case}_{ips[0]}')
    local_run(log_file, cmd, nsys, python_cmd=python_cmd)


def choose_and_run_test_cases(filepath, handler, port=None, nsys='', python_cmd='python3', workspace='/workspace'):
    with open(filepath, newline='', encoding='UTF-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            print('{}: {}'.format(i, row['case']))

    n = input("Please Choose a number, 'a' for all, otherwise quit: ")
    if n == '':
        print('quit')
        exit()

    assert n != None
    if n != "a" and n != "":
        n = int(n)
        # assert n >= 0 and n < len(df), "Invalid number!"

    with open(filepath, newline="", encoding='UTF-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if isinstance(n, int) and n != i and n != 'a':
                continue
            args, env, test_case = handler(row)
            gpt_run(args, env, test_case, port, nsys, python_cmd, workspace)
